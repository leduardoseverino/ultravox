import argparse
import asyncio
import datetime
import json
import logging
import os
import signal
import sys
import urllib.parse
from typing import Any, AsyncGenerator, Awaitable, Literal

import aiohttp
import numpy as np
import pyee.asyncio
import sounddevice
from websockets import exceptions as ws_exceptions
from websockets.asyncio import client as ws_client

VOICE_NAME = "Keren-Brazilian-Portuguese"


class LocalAudioSink:
    """
    A sink for audio. Buffered audio is played using the default audio device.

    Args:
        sample_rate: The sample rate to use for audio playback. Defaults to 48kHz.
    """

    def __init__(self, sample_rate: int = 48000) -> None:
        self._sample_rate = sample_rate
        self._buffer: bytearray = bytearray()

        def callback(outdata: np.ndarray, frame_count, time, status):
            output_frame_size = len(outdata) * 2
            next_frame = self._buffer[:output_frame_size]
            self._buffer[:] = self._buffer[output_frame_size:]
            if len(next_frame) < output_frame_size:
                next_frame += b"\x00" * (output_frame_size - len(next_frame))
            outdata[:] = np.frombuffer(next_frame, dtype="int16").reshape(
                (frame_count, 1)
            )

        self._stream = sounddevice.OutputStream(
            samplerate=sample_rate,
            channels=1,
            callback=callback,
            device=None,
            dtype="int16",
            blocksize=sample_rate // 100,
        )
        self._stream.start()
        if not self._stream.active:
            raise RuntimeError("Failed to start streaming output audio")

    def write(self, chunk: bytes) -> None:
        """Writes audio data (expected to be in 16-bit PCM format) to this sink's buffer."""
        self._buffer.extend(chunk)

    def drop_buffer(self) -> None:
        """Drops all audio data in this sink's buffer, ending playback until new data is written."""
        self._buffer.clear()

    async def close(self) -> None:
        if self._stream:
            self._stream.close()


class LocalAudioSource:
    """
    A source for audio data that reads from the default microphone. Audio data in
    16-bit PCM format is available as an AsyncGenerator via the `stream` method.

    Args:
        sample_rate: The sample rate to use for audio recording. Defaults to 48kHz.
    """

    def __init__(self, sample_rate=48000):
        self._sample_rate = sample_rate

    async def stream(self) -> AsyncGenerator[bytes, None]:
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def callback(indata: np.ndarray, frame_count, time, status):
            loop.call_soon_threadsafe(queue.put_nowait, indata.tobytes())

        stream = sounddevice.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            callback=callback,
            device=None,
            dtype="int16",
            blocksize=self._sample_rate // 100,
        )
        with stream:
            if not stream.active:
                raise RuntimeError("Failed to start streaming input audio")
            while True:
                yield await queue.get()


class WebsocketVoiceSession(pyee.asyncio.AsyncIOEventEmitter):
    """A websocket-based voice session that connects to an Ultravox call. The session continuously
    streams audio in and out and emits events for state changes and agent messages."""

    def __init__(self, join_url: str):
        super().__init__()
        self._state: Literal["idle", "listening", "thinking", "speaking"] = "idle"
        self._pending_output = ""
        self._url = join_url
        self._socket = None
        self._receive_task: asyncio.Task | None = None
        self._send_audio_task = asyncio.create_task(
            self._pump_audio(LocalAudioSource())
        )
        self._sink = LocalAudioSink()

    async def start(self):
        logging.info(f"Connecting to {self._url}")
        self._socket = await ws_client.connect(self._url)
        self._receive_task = asyncio.create_task(self._socket_receive(self._socket))

    async def _socket_receive(self, socket: ws_client.ClientConnection):
        try:
            async for message in socket:
                await self._on_socket_message(message)
        except asyncio.CancelledError:
            logging.info("socket cancelled")
        except ws_exceptions.ConnectionClosedOK:
            logging.info("socket closed ok")
        except ws_exceptions.ConnectionClosedError as e:
            self.emit("error", e)
            return
        logging.info("socket receive done")
        self.emit("ended")

    async def stop(self):
        """End the session, closing the connection and ending the call."""
        logging.info("Stopping...")
        await _async_close(
            self._sink.close(),
            self._socket.close() if self._socket else None,
            _async_cancel(self._send_audio_task, self._receive_task),
        )
        if self._state != "idle":
            self._state = "idle"
            self.emit("state", "idle")

    async def _on_socket_message(self, payload: str | bytes):
        if isinstance(payload, bytes):
            self._sink.write(payload)
            return
        elif isinstance(payload, str):
            msg = json.loads(payload)
            await self._handle_data_message(msg)

    async def _handle_data_message(self, msg: dict[str, Any]):
        match msg["type"]:
            case "playback_clear_buffer":
                self._sink.drop_buffer()
            case "state":
                if msg["state"] != self._state:
                    self._state = msg["state"]
                    self.emit("state", msg["state"])
            case "transcript":
                # This is lazy handling of transcripts. See the WebRTC client SDKs
                # for a more robust implementation.
                if msg["role"] != "agent":
                    return  # Ignore user transcripts
                if msg.get("text", None):
                    self._pending_output = msg["text"]
                    self.emit("output", msg["text"], msg["final"])
                else:
                    self._pending_output += msg.get("delta", "")
                    self.emit("output", self._pending_output, msg["final"])
                if msg["final"]:
                    self._pending_output = ""
            case "client_tool_invocation":
                await self._handle_client_tool_call(
                    msg["toolName"], msg["invocationId"], msg["parameters"]
                )
            case "debug":
                logging.info(f"debug: {msg['message']}")
            case _:
                logging.warning(f"Unhandled message type: {msg['type']}")

    async def _handle_client_tool_call(
        self, tool_name: str, invocation_id: str, parameters: dict[str, Any]
    ):
        logging.info(f"client tool call: {tool_name}")
        response: dict[str, str] = {
            "type": "client_tool_result",
            "invocationId": invocation_id,
        }
        if tool_name == "getSecretMenu":
            menu = [
                {
                    "date": datetime.date.today().isoformat(),
                    "items": [
                        {
                            "name": "Banana Smoothie",
                            "price": "$4.99",
                        },
                        {
                            "name": "Butter Pecan Ice Cream (one scoop)",
                            "price": "$2.99",
                        },
                    ],
                },
            ]
            response["result"] = json.dumps(menu)
        else:
            response["errorType"] = "undefined"
            response["errorMessage"] = f"Unknown tool: {tool_name}"
        await self._socket.send(json.dumps(response))

    async def _pump_audio(self, source: LocalAudioSource):
        async for chunk in source.stream():
            if self._socket is None:
                continue
            await self._socket.send(chunk)


async def _async_close(*awaitables_or_none: Awaitable | None):
    coros = [coro for coro in awaitables_or_none if coro is not None]
    if coros:
        maybe_exceptions = await asyncio.shield(
            asyncio.gather(*coros, return_exceptions=True)
        )
        non_cancelled_exceptions = [
            exc
            for exc in maybe_exceptions
            if isinstance(exc, Exception)
            and not isinstance(exc, asyncio.CancelledError)
        ]
        if non_cancelled_exceptions:
            to_report = (
                non_cancelled_exceptions[0]
                if len(non_cancelled_exceptions) == 1
                else ExceptionGroup("Multiple failures", non_cancelled_exceptions)
            )
            logging.warning("Error during _async_close", exc_info=to_report)


async def _async_cancel(*tasks_or_none: asyncio.Task | None):
    tasks = [task for task in tasks_or_none if task is not None and task.cancel()]
    await _async_close(*tasks)


async def _get_join_url() -> str:
    """Creates a new call, returning its join URL."""
    
    # Remove or comment out this line to stop printing the API key
    # print(f"Using API key: {os.getenv('ULTRAVOX_API_KEY', 'Not set')}")

    target = "https://api.ultravox.ai/api/calls"
    if args.prior_call_id:
        target += f"?priorCallId={args.prior_call_id}"
    async with aiohttp.ClientSession() as session:
        headers = {"X-API-Key": f"{os.getenv('ULTRAVOX_API_KEY', None)}"}
        system_prompt = args.system_prompt
        selected_tools = []
        if args.secret_menu:
            system_prompt += "\n\nThere is also a secret menu that changes daily. If the user asks about it, use the getSecretMenu tool to look up today's secret menu items."
            selected_tools.append(
                {
                    "temporaryTool": {
                        "modelToolName": "getSecretMenu",
                        "description": "Looks up today's secret menu items.",
                        "client": {},
                    },
                }
            )
        body = {
            "systemPrompt": system_prompt,
            "temperature": args.temperature,
            "medium": {
                "serverWebSocket": {
                    "inputSampleRate": 48000,
                    "outputSampleRate": 48000,
                    # Buffer up to 30s of audio client-side. This won't impact
                    # interruptions because we handle playback_clear_buffer above.
                    "clientBufferSizeMs": 30000,
                }
            },
        }
        if args.voice:
            body["voice"] = args.voice
        if selected_tools:
            body["selectedTools"] = selected_tools
        if args.initial_output_text:
            body["initialOutputMedium"] = "MESSAGE_MEDIUM_TEXT"
        if args.user_speaks_first:
            body["firstSpeaker"] = "FIRST_SPEAKER_USER"

        logging.info(f"Creating call with body: {body}")
        async with session.post(target, headers=headers, json=body) as response:
            response.raise_for_status()
            response_json = await response.json()
            join_url = response_json["joinUrl"]
            join_url = _add_query_param(
                join_url, "apiVersion", str(args.api_version or 1)
            )
            if args.experimental_messages:
                join_url = _add_query_param(
                    join_url, "experimentalMessages", args.experimental_messages
                )
            return join_url


def _add_query_param(url: str, key: str, value: str) -> str:
    url_parts = list(urllib.parse.urlparse(url))
    query = dict(urllib.parse.parse_qsl(url_parts[4]))
    query.update({key: value}) 
    url_parts[4] = urllib.parse.urlencode(query)
    return urllib.parse.urlunparse(url_parts)


async def main():
    join_url = await _get_join_url()
    client = WebsocketVoiceSession(join_url)
    done = asyncio.Event()
    loop = asyncio.get_running_loop()

    # Platform-specific signal handling
    if os.name == 'posix':  # Unix-like systems (Linux, macOS)
        loop.add_signal_handler(signal.SIGINT, lambda: done.set())
        loop.add_signal_handler(signal.SIGTERM, lambda: done.set())
    else:  # Windows
        signal.signal(signal.SIGINT, lambda s, f: done.set())
        signal.signal(signal.SIGTERM, lambda s, f: done.set())

    @client.on("state")
    async def on_state(state):
        if state == "listening":
            # Used to prompt the user to speak
            print("User:  ", end="\r")
        elif state == "thinking":
            print("Agent: ", end="\r")

    @client.on("output")
    async def on_output(text, final):
        display_text = f"{text.strip()}"
        print("Agent: " + display_text, end="\n" if final else "\r")

    @client.on("error")
    async def on_error(error):
        logging.exception("Client error", exc_info=error)
        print(f"Error: {error}")
        done.set()

    @client.on("ended")
    async def on_ended():
        print("Session ended")
        done.set()

    await client.start()
    await done.wait()
    await client.stop()


if __name__ == "__main__":
    api_key = os.getenv("ULTRAVOX_API_KEY", None)
    if not api_key:
        raise ValueError("Please set your ULTRAVOX_API_KEY environment variable")

    parser = argparse.ArgumentParser(prog="websocket_client.py")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose session information"
    )
    parser.add_argument(
        "--very-verbose", "-vv", action="store_true", help="Show debug logs too"
    )

    parser.add_argument(
        "--voice",
        "-V",
        type=str,
        default=VOICE_NAME,
        help="Name (or id) of voice to use"
    )
    parser.add_argument(
        "--system-prompt",
        "-s",
        type=str,
        default=f"""Você é um atendente por telefone da Skyone, especializada em soluções digitais como o Skyone Studio e o Autosky. A hora local atualmente é: ${datetime.datetime.now().isoformat()}

O usuário está conversando com você por voz no telefone, e sua resposta será lida em voz alta com tecnologia de síntese de voz (TTS) realista.

Siga todas as direções abaixo ao elaborar sua resposta:
	1.	Inicie perguntando se o cliente precisa de ajuda para identificar qual solução digital atende melhor ao seu cenário atual.
	•	Se sim, faça perguntas como:
	•	“Você está buscando algo para automatizar processos, integrar dados, aplicar inteligência artificial ou tudo isso junto?”
	•	“Você já possui alguma infraestrutura, como um banco de dados ou ferramenta de BI?”
	•	“Com que frequência você precisa acessar ou atualizar esses dados?”
	•	Ajude o cliente a identificar a combinação ideal entre recursos do Skyone Studio e do Autosky.
	•	Se o cliente já souber o que precisa, peça que diga o nome do produto ou funcionalidade.
	2.	Use linguagem natural e conversacional, com frases curtas, palavras simples e tom direto.
	•	Seja conciso e relevante: a maioria das respostas deve ter uma ou duas frases, a não ser que o cliente peça mais detalhes.
	•	Use expressões comuns para facilitar a compreensão, como “então tá bom”, “certo”, “beleza”, “entendi”.
	3.	Mantenha a conversa fluindo:
	•	Esclareça dúvidas com perguntas simples, sem assumir nada por conta própria.
	•	Não tente encerrar a conversa. Nunca diga “tchau” ou “obrigado por ligar”.
	•	Se o cliente estiver só explorando, incentive com curiosidade: “Legal! Quer saber o que o Studio faz com IA, por exemplo?”
	4.	Esta é uma conversa por voz:
	•	Evite listas, termos técnicos ou frases difíceis.
	•	Use números por extenso, como “doze mil” em vez de “12000”.
	•	Se algo soar confuso, peça para o cliente explicar de outro jeito. Não há erro de digitação, apenas de compreensão.

Script para atendimento:
	1.	Após entender a necessidade, apresente soluções da Skyone com clareza:
	•	Fale de forma geral no início, por exemplo: “O Skyone Studio pode te ajudar com automações, integrações de sistemas e até agentes com IA para tomar decisões automáticas.”
	•	À medida que o cliente mostra interesse, vá detalhando: “Esse tipo de fluxo pode ser construído visualmente, com regras de negócio e até IA conectada.”
	2.	Quando o cliente mostrar intenção:
	•	Confirme o que ele deseja explorar.
	•	Se ele estiver olhando algo simples, pergunte se quer conhecer soluções mais robustas.
	•	Se ele quiser algo mais avançado, veja se também quer conhecer opções mais práticas.
	3.	Não mencione valores, descontos, planos ou comparativos de preços.
	4.	Caso o cliente pergunte algo que não está relacionado à Skyone ou aos seus produtos, responda com:
	•	“Um… isso é a Skyone, a gente trabalha com soluções digitais, tá bom?”
	5.	Se o cliente disser “obrigado”, diga “De nada”.
	6.	Se ele pedir recomendações, não leia todos os produtos. Em vez disso, dê alguns exemplos, como:
	•	“Olha, se você quer começar a automatizar, posso te mostrar o Studio com IA.”
	•	“Se estiver buscando organizar e centralizar os dados, o Lakehouse é ideal.”
"""
,
        help="System prompt to use when creating the call",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature to use when creating the call",
    )
    parser.add_argument(
        "--secret-menu",
        action="store_true",
        help="Adds prompt and client-implemented tool for a secret menu. For use with the default system prompt.",
    )
    parser.add_argument(
        "--experimental-messages",
        type=str,
        help="Enables the specified experimental messages (e.g. 'debug' which should be used with -v)",
    )
    parser.add_argument(
        "--prior-call-id",
        type=str,
        help="Allows setting priorCallId during start call",
    )
    parser.add_argument(
        "--user-speaks-first",
        action="store_true",
        help="If set, sets FIRST_SPEAKER_USER",
    )
    parser.add_argument(
        "--initial-output-text",
        action="store_true",
        help="Sets the initial_output_medium to text",
    )
    parser.add_argument(
        "--api-version",
        type=int,
        help="API version to set when creating the call.",
    )

    args = parser.parse_args()
    if args.very_verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    asyncio.run(main())

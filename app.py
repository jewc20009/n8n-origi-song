import asyncio
from fastapi import FastAPI, Request, Header, HTTPException
from flowise import Flowise, PredictionData
import json
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

app = FastAPI()

# Definimos el BaseURL fijo dentro del código
BASE_URL = "https://flowise-liy3.onrender.com"

async def stream_response(client, question: str, session_id: str, ChatflowId: str) -> AsyncGenerator[str, None]:
    """Genera la respuesta de streaming de Flowise y la envía como fragmentos"""
    completion = client.create_prediction(
        PredictionData(
            chatflowId=ChatflowId,  # Usamos session_id como chatflowId
            question=question,
            streaming=True,
            overrideConfig={"SessionId": session_id}  # SessionId
        )
    )

    for chunk in completion:
        try:
            if isinstance(chunk, str):
                chunk_data = json.loads(chunk)
                if "data" in chunk_data:
                    for data in chunk_data["data"]:
                        if isinstance(data, dict):
                            message = data.get("messages", [])[0] if isinstance(data.get("messages"), list) and data.get("messages") else None
                            if message:
                                response_chunk = json.dumps({
                                    "choices": [{
                                        "delta": {"content": message},
                                        "finish_reason": None
                                    }]
                                }) + "\n\n"
                                yield response_chunk
        except json.JSONDecodeError:
            continue

    # Cuando el stream termina, envía el final
    yield json.dumps({
        "choices": [{
            "delta": {},
            "finish_reason": "stop"
        }]
    }) + "\n\n"

@app.post("/v1/chat/completions")
async def chat_completion(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

    # Extraer el session_id del header Authorization
    session_id = authorization.replace("Bearer ", "")

    body = await request.json()
    ChatflowId = body.get("model")
    question = body["messages"][-1]["content"]

    # Configurar el cliente Flowise con el base_url fijo
    client = Flowise(base_url=BASE_URL, api_key=session_id)  # session_id se usa como api_key

    # Enviar la respuesta de streaming
    return StreamingResponse(stream_response(client, question, session_id, ChatflowId), media_type="application/json")

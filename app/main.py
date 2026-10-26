import base64
import io
import secrets
import wave
from pathlib import Path
from uuid import uuid4
import requests
import time

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from groq import Groq
from langchain_groq import ChatGroq

from app.config import settings
from app.agent import graph

ROOT=Path(__file__).resolve().parent.parent
app=FastAPI(title='Medical Agent Educational API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=settings.allowed_origins.split(','),allow_credentials=False,allow_methods=['GET','POST'],allow_headers=['Authorization','Content-Type'])
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')


class ChatRequest(BaseModel):
    message: str = Field(min_length=1,max_length=4000)
    history: list[dict[str,str]] = Field(default_factory=list,max_length=10)
    image_data: str | None = None

def auth(authorization: str | None):
    if settings.api_token and not secrets.compare_digest(authorization or '',f'Bearer {settings.api_token}'):
        raise HTTPException(status_code=401,detail='Invalid API token')


def inspect_image(data):
    if not data:
        return ''
    try:
        header,payload=data.split(',',1)
        mime=header.split(';')[0].split(':')[1]
        raw=base64.b64decode(payload,validate=True)
        if mime not in ('image/jpeg','image/png','image/webp') or len(raw)>5_000_000:
            raise ValueError('Only JPEG/PNG/WebP under 5MB supported')
    except (ValueError,IndexError) as exc:
        raise HTTPException(status_code=400,detail=f'Invalid image: {exc}')

    client=ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_vision_model,
        timeout=45,
    )
    result=client.invoke([
        {
            'role':'user',
            'content':[
                {
                    'type':'text',
                    'text':'Describe only objectively visible medical-relevant features. Do not diagnose, estimate severity, or assert absence of illness. If image unclear, say so.'
                },
                {
                    'type':'image_url',
                    'image_url':{'url':data}
                }
            ]
        }
    ])
    return result.content or ''


@app.get('/',response_class=HTMLResponse)
def home():
    return (ROOT/'templates'/'index.html').read_text(encoding='utf-8')

@app.get('/api/health')
def health():
    return {'status':'ok','configured':bool(settings.groq_api_key and settings.pinecone_api_key),'tavily':bool(settings.tavily_api_key)}



@app.post('/api/chat')
def chat(
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Header(default=None, alias='X-Conversation-ID')
):
    auth(authorization)

    if not settings.groq_api_key or not settings.pinecone_api_key:
        raise HTTPException(503,'Set GROQ_API_KEY and PINECONE_API_KEY in .env')

    history=[x for x in payload.history if x.get('role') in ('user','assistant') and isinstance(x.get('content'),str)][:10]

    # SQLite persistent memory: reuse the same conversation thread
    thread_id = conversation_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        observation=inspect_image(payload.image_data)

        state=graph.invoke(
            {
                'question':payload.message,
                'history':history,
                'image_description':observation,
                'kb':[],
                'web':[],
                'answer':'',
                'route':''
            },
            config=config
        )

        return {
            'answer':state['answer'],
            'sources': [{'type':'pdf','page':x['page'],'score':x['score']} for x in state['kb']]+[{'type':'web','url':x['url'],'title':x['title']} for x in state['web']],
            'tools': ['pinecone']+(['tavily'] if state['route']=='web' and settings.tavily_api_key else [])+(['vision'] if observation else []),
            'conversation_id':thread_id
        }

    except HTTPException:
        raise
    except Exception as exc:
        print('Chat error:',repr(exc),flush=True)
        raise HTTPException(502,'AI service failed. Check server logs, credentials, Pinecone index, and connectivity.')



@app.post('/api/transcribe')
async def transcribe(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None)
):
    auth(authorization)

    data = await file.read(10_000_001)

    if len(data) > 10_000_000:
        raise HTTPException(413, 'Audio exceeds 10 MB')

    if not data:
        raise HTTPException(400, 'Empty audio file')

    audio = io.BytesIO(data)
    audio.name = file.filename or 'voice.webm'

    try:
        client = Groq(api_key=settings.groq_api_key)

        result = client.audio.transcriptions.create(
            model='whisper-large-v3',
            file=(audio.name, audio, file.content_type or 'audio/webm'),
            response_format='json'
        )

        return {'text': result.text}

    except Exception as exc:
        print('Transcription error:', repr(exc), flush=True)
        raise HTTPException(502, 'Speech transcription unavailable')




@app.post('/api/speak')
def speak(payload: dict, authorization: str | None = Header(default=None)):
    import io
    import time
    import wave
    import requests

    auth(authorization)

    text = str(payload.get('text', '')).strip()[:3000]

    if not text:
        raise HTTPException(status_code=400, detail='Missing text')

    if not settings.groq_api_key:
        raise HTTPException(status_code=503, detail='GROQ_API_KEY is missing')

    # Split the answer into chunks of at most 200 characters.
    chunks = []
    while text:
        if len(text) <= 200:
            chunks.append(text)
            break

        split_at = text.rfind(' ', 0, 200)
        if split_at <= 0:
            split_at = 200

        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    try:
        audio_parts = []
        audio_params = None

        for chunk in chunks:
            response = None

            # Retry rate-limit errors up to 3 attempts.
            for attempt in range(3):
                response = requests.post(
                    'https://api.groq.com/openai/v1/audio/speech',
                    headers={
                        'Authorization': f'Bearer {settings.groq_api_key}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'canopylabs/orpheus-v1-english',
                        'voice': 'troy',
                        'input': chunk,
                        'response_format': 'wav'
                    },
                    timeout=90
                )

                if response.status_code != 429:
                    break

                if attempt < 2:
                    try:
                        wait = float(
                            response.headers.get('retry-after', '3')
                        )
                    except (ValueError, TypeError):
                        wait = 3

                    # Avoid excessively long waits.
                    time.sleep(min(max(wait, 2), 10))

            if not response.ok:
                print(
                    'Groq TTS error:',
                    response.status_code,
                    response.text[:1000],
                    flush=True
                )

                raise HTTPException(
                    status_code=502,
                    detail=(
                        f'Groq TTS API error '
                        f'({response.status_code}): '
                        f'{response.text[:500]}'
                    )
                )

            with wave.open(io.BytesIO(response.content), 'rb') as wav_file:
                params = wav_file.getparams()

                if audio_params is None:
                    audio_params = params
                elif (
                    params.nchannels != audio_params.nchannels
                    or params.sampwidth != audio_params.sampwidth
                    or params.framerate != audio_params.framerate
                    or params.comptype != audio_params.comptype
                ):
                    raise ValueError(
                        'Groq returned incompatible WAV audio formats'
                    )

                frames = wav_file.readframes(wav_file.getnframes())
                audio_parts.append(frames)

        if not audio_parts or audio_params is None:
            raise ValueError('No audio was generated')

        # Combine audio frames into one valid WAV file.
        output = io.BytesIO()

        with wave.open(output, 'wb') as wav_output:
            wav_output.setnchannels(audio_params.nchannels)
            wav_output.setsampwidth(audio_params.sampwidth)
            wav_output.setframerate(audio_params.framerate)
            wav_output.setcomptype(
                audio_params.comptype,
                audio_params.compname
            )

            for part in audio_parts:
                wav_output.writeframes(part)

        return Response(
            content=output.getvalue(),
            media_type='audio/wav',
            headers={
                'Content-Disposition': 'inline; filename=speech.wav'
            }
        )

    except HTTPException:
        raise
    except requests.RequestException as exc:
        print('Groq request error:', repr(exc), flush=True)
        raise HTTPException(
            status_code=502,
            detail='Could not connect to Groq TTS service'
        )
    except Exception as exc:
        print('Speech synthesis error:', repr(exc), flush=True)
        raise HTTPException(
            status_code=502,
            detail=f'Speech synthesis failed: {str(exc)}'
        )

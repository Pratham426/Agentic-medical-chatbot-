
const $ = id => document.getElementById(id);

let history = [];
let imageData = null;
let recorder = null;
let audioChunks = [];
let cameraStream = null;
let lastAnswer = '';
let currentAudio = null;
let currentAudioUrl = null;

const headers = () => {
    const h = {};
    const token = $('token').value.trim();

    if (token) {
        h.Authorization = `Bearer ${token}`;
    }

    return h;
};

function bubble(text, kind = 'bot', sources = []) {
    const el = document.createElement('article');
    el.className = kind;
    el.textContent = text;

    if (sources.length) {
        const s = document.createElement('div');
        s.className = 'sources';
        s.textContent = 'Sources: ' + sources
            .slice(0, 8)
            .map(x => x.type === 'pdf' ? `PDF page ${x.page}` : x.url)
            .join(' | ');

        el.appendChild(s);
    }

    $('messages').appendChild(el);
    $('messages').scrollTop = $('messages').scrollHeight;

    return el;
}

async function check() {
    try {
        const r = await fetch('/api/health');
        const d = await r.json();

        $('status').textContent = d.configured
            ? '● Connected'
            : '⚠ Configure API keys';
    } catch {
        $('status').textContent = '● Server offline';
    }
}

check();

function setImage(file) {
    if (!file) return;

    if (
        !['image/png', 'image/jpeg', 'image/webp'].includes(file.type) ||
        file.size > 5e6
    ) {
        alert('Use PNG/JPEG/WebP under 5MB');
        return;
    }

    const reader = new FileReader();

    reader.onload = () => {
        imageData = reader.result;
        $('thumbnail').src = imageData;
        $('preview').hidden = false;
    };

    reader.readAsDataURL(file);
}

$('image').addEventListener('change', e => {
    setImage(e.target.files[0]);
});

$('remove').onclick = () => {
    imageData = null;
    $('preview').hidden = true;
    $('image').value = '';
};

$('chatForm').addEventListener('submit', async e => {
    e.preventDefault();

    const message = $('message').value.trim();
    if (!message) return;

    const prior = history.slice(-8);

    bubble(message, 'user');
    $('message').value = '';
    $('send').disabled = true;

    const pending = bubble('Thinking…');

    try {
        const r = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                ...headers(),
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                history: prior,
                image_data: imageData
            })
        });

        const d = await r.json();

        if (!r.ok) {
            throw new Error(d.detail || `Chat request failed (${r.status})`);
        }

        pending.textContent = d.answer;
        lastAnswer = d.answer || '';

        if (d.sources?.length) {
            const src = document.createElement('div');
            src.className = 'sources';
            src.textContent = 'Sources: ' + d.sources
                .slice(0, 8)
                .map(x => x.type === 'pdf' ? `PDF page ${x.page}` : x.url)
                .join(' | ');

            pending.appendChild(src);
        }

        history.push(
            { role: 'user', content: message },
            { role: 'assistant', content: d.answer }
        );

        $('remove').click();

    } catch (err) {
        pending.textContent = 'Error: ' + err.message;
        console.error('Chat error:', err);
    } finally {
        $('send').disabled = false;
        $('messages').scrollTop = $('messages').scrollHeight;
    }
});

// Voice input: speech-to-text
$('mic').onclick = async () => {
    if (recorder && recorder.state === 'recording') {
        recorder.stop();
        $('mic').textContent = '🎤 Voice';
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });

        audioChunks = [];
        recorder = new MediaRecorder(stream);

        recorder.ondataavailable = e => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        recorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());

            const blob = new Blob(audioChunks, {
                type: recorder.mimeType
            });

            const data = new FormData();
            data.append('file', blob, 'voice.webm');

            $('mic').disabled = true;

            try {
                const r = await fetch('/api/transcribe', {
                    method: 'POST',
                    headers: headers(),
                    body: data
                });

                const out = await r.json();

                if (!r.ok) {
                    throw new Error(out.detail || `Transcription failed (${r.status})`);
                }

                $('message').value = out.text || '';

            } catch (err) {
                alert('Voice input failed: ' + err.message);
            } finally {
                $('mic').disabled = false;
            }
        };

        recorder.start();
        $('mic').textContent = '⏹ Stop recording';

    } catch (err) {
        alert('Microphone unavailable: ' + err.message);
    }
};

// Read Answer: text-to-speech
$('listen').onclick = async () => {
    if (!lastAnswer) {
        alert('Ask a question first');
        return;
    }

    $('listen').disabled = true;

    try {
        // Stop any previous audio playback.
        if (currentAudio) {
            currentAudio.pause();
            currentAudio = null;
        }

        if (currentAudioUrl) {
            URL.revokeObjectURL(currentAudioUrl);
            currentAudioUrl = null;
        }

        const r = await fetch('/api/speak', {
            method: 'POST',
            headers: {
                ...headers(),
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: lastAnswer
            })
        });

        if (!r.ok) {
            const errorText = await r.text();

            throw new Error(
                `HTTP ${r.status}: ${errorText}`
            );
        }

        const blob = await r.blob();

        if (!blob.size) {
            throw new Error('The server returned empty audio');
        }

        currentAudioUrl = URL.createObjectURL(blob);
        currentAudio = new Audio(currentAudioUrl);

        currentAudio.onended = () => {
            if (currentAudioUrl) {
                URL.revokeObjectURL(currentAudioUrl);
                currentAudioUrl = null;
            }

            currentAudio = null;
        };

        currentAudio.onerror = () => {
            console.error('Audio playback failed', currentAudio?.error);
            alert('Audio was received, but the browser could not play it.');
        };

        await currentAudio.play();

    } catch (err) {
        console.error('Read Answer error:', err);
        alert('Read Answer failed: ' + err.message);
    } finally {
        $('listen').disabled = false;
    }
};

// Camera
$('camera').onclick = async () => {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });

        $('video').srcObject = cameraStream;
        $('video').hidden = false;
        $('cameraActions').hidden = false;

    } catch (err) {
        alert('Camera unavailable: ' + err.message);
    }
};

$('closeCamera').onclick = () => {
    cameraStream?.getTracks().forEach(t => t.stop());
    cameraStream = null;

    $('video').srcObject = null;
    $('video').hidden = true;
    $('cameraActions').hidden = true;
};

$('snap').onclick = () => {
    const v = $('video');
    const c = $('canvas');

    c.width = v.videoWidth;
    c.height = v.videoHeight;

    c.getContext('2d').drawImage(v, 0, 0);

    c.toBlob(blob => {
        if (blob) {
            setImage(new File([blob], 'camera.jpg', {
                type: 'image/jpeg'
            }));
        }
    }, 'image/jpeg', 0.85);

    $('closeCamera').click();
};

let mediaRecorder;
let audioChunks = [];

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = event => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      audioChunks = [];

      const formData = new FormData();
      formData.append('audio', blob, 'clip.webm');

      const res = await fetch('https://mood-api-bpreddy.onrender.com/analyze-mood', {
        method: 'POST',
        body: formData
      });

      const result = await res.json();
      document.getElementById('mood').innerText = `Mood: ${result.mood}`;
    };

    mediaRecorder.start();

    setInterval(() => {
      if (mediaRecorder.state === 'recording') {
        mediaRecorder.stop();  // analyze
        mediaRecorder.start(); // start next chunk
      }
    }, 30000); // every 30 secs
  } catch (err) {
    alert('Microphone permission denied.');
  }
}

window.onload = () => {
  startRecording();
};

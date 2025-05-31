// ===============================
// script.js
// ===============================

let mediaRecorder;
let audioChunks = [];
let intervalID = null;
let moodDisplay = document.getElementById("moodDisplay");

function startRecordingCycle() {
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks, { type: 'audio/wav' });
        audioChunks = [];

        const formData = new FormData();
        formData.append('audio', blob, 'sample.wav');
        const username = localStorage.getItem("username");
        formData.append('username', username);

        fetch('/analyze', {
          method: 'POST',
          body: formData
        })
          .then(response => response.json())
          .then(data => {
            displayMood(data.mood);
          })
          .catch(error => {
            console.error('Error sending audio:', error);
          });
      };

      // Start recording every 30 seconds
      intervalID = setInterval(() => {
        if (mediaRecorder.state === "recording") {
          mediaRecorder.stop();
        }
        mediaRecorder.start();
        setTimeout(() => mediaRecorder.stop(), 5000);  // record for 5 seconds
      }, 30000);

      // Initial start
      mediaRecorder.start();
      setTimeout(() => mediaRecorder.stop(), 5000);

    })
    .catch(error => {
      console.error('Microphone error:', error);
    });
}

function displayMood(mood) {
  moodDisplay.innerText = `Current Mood: ${mood}`;
  const imageDisplay = document.getElementById("imageDisplay");

  if (mood === "Happy") {
    imageDisplay.innerHTML = "❤️";
  } else {
    fetch(`/images/${mood.toLowerCase()}`)
      .then(response => response.json())
      .then(images => {
        imageDisplay.innerHTML = "";
        images.forEach(src => {
          const img = document.createElement("img");
          img.src = src;
          img.className = "mood-img";
          imageDisplay.appendChild(img);
        });
      });
  }
}

window.onload = () => {
  const loggedIn = localStorage.getItem("loggedIn");
  if (loggedIn === "true") {
    startRecordingCycle();
  }
};

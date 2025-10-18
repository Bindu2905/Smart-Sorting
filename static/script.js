const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const openBtn = document.getElementById("openCamBtn");
const captureBtn = document.getElementById("captureBtn");
const closeBtn = document.getElementById("closeCamBtn");
const statusDiv = document.getElementById("status");

let stream;

// ✅ Open camera
openBtn.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    document.getElementById("cameraWrap").classList.remove("hidden");
    statusDiv.textContent = "Camera opened. Say 'take image' to capture.";
    speak("Camera is now open.");
  } catch (err) {
    console.error("Camera error:", err);
    speak("Unable to access camera.");
  }
});

// ✅ Capture image & send for prediction
captureBtn.addEventListener("click", async () => {
  if (!stream) {
    speak("Please open the camera first.");
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  const imageData = canvas.toDataURL("image/jpeg");

  statusDiv.textContent = "Analyzing image, please wait...";
  speak("Analyzing image, please wait...");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageData })
    });

    const data = await response.json();

    // ✅ Display + Voice Output
    statusDiv.textContent = "🧾 Result: " + data.result;
    speak("The fruit is " + data.result);
  } catch (err) {
    console.error("Prediction error:", err);
    speak("Error analyzing the image.");
  }
});

// ✅ Close camera
closeBtn.addEventListener("click", () => {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
    document.getElementById("cameraWrap").classList.add("hidden");
    statusDiv.textContent = "Camera closed.";
    speak("Camera closed.");
  }
});

// ✅ Text-to-speech helper
function speak(text) {
  const u = new SpeechSynthesisUtterance(text);
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}
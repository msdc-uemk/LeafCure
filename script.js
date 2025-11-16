document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("fileInput");
    const uploadBtn = document.getElementById("uploadBtn");
    const openCameraBtn = document.getElementById("openCameraBtn");
    const cameraModal = document.getElementById("cameraModal");
    const closeCameraBtn = document.querySelector(".close-btn");
    const video = document.getElementById("video");
    const captureBtn = document.getElementById("captureBtn");
    const resultDisplay = document.getElementById("resultDisplay");
    const capturedImage = document.getElementById("capturedImage");

    // ✅ Specific result elements
    const resultCrop = document.getElementById("resultCrop");
    const resultDisease = document.getElementById("resultDisease");
    const confidencePercent = document.getElementById("confidencePercent");
    const confidenceBar = document.getElementById("confidenceBar");

    // ==============================
    // Helper: Send file to Flask backend
    // ==============================
    async function analyzeImage(file) {
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });
            const data = await response.json();

            if (data.error) {
                alert(data.error);
                return;
            }

            // ✅ Update results
            resultCrop.textContent = `${data.crop.charAt(0).toUpperCase() + data.crop.slice(1)} Leaf`;
            resultDisease.textContent = `${data.disease} detected`;

            // ✅ Real confidence %
            const confidence = Math.round(data.confidence * 100); // e.g. 0.9273 → 93
            confidencePercent.textContent = `${confidence}%`;

            // Animate confidence bar up to real value
            let width = 0;
            confidenceBar.style.width = "0%";
            const interval = setInterval(() => {
                width += 2;
                confidenceBar.style.width = `${width}%`;
                if (width >= confidence) clearInterval(interval);
            }, 30);

            // Show result section
            resultDisplay.classList.remove("hidden");
            resultDisplay.scrollIntoView({ behavior: "smooth" });

        } catch (err) {
            console.error("Prediction error:", err);
            alert("Something went wrong during analysis.");
        }
    }

    // ==============================
    // File upload handler
    // ==============================
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                capturedImage.src = event.target.result;
            };
            reader.readAsDataURL(file);
            analyzeImage(file);
        }
    });

    // Upload button opens file dialog
    uploadBtn.addEventListener("click", () => fileInput.click());

    // ==============================
    // Camera modal handlers
    // ==============================
    openCameraBtn.addEventListener("click", async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            cameraModal.style.display = "block";
        } catch (err) {
            console.error("Camera error:", err);
            alert("Could not access camera.");
        }
    });

    closeCameraBtn.addEventListener("click", () => {
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
        cameraModal.style.display = "none";
    });

    captureBtn.addEventListener("click", () => {
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0);

        // Stop camera
        video.srcObject.getTracks().forEach(track => track.stop());
        cameraModal.style.display = "none";

        canvas.toBlob((blob) => {
            if (blob) {
                capturedImage.src = URL.createObjectURL(blob);
                analyzeImage(blob);
            }
        }, "image/jpeg");
    });
});

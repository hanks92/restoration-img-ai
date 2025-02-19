function uploadImage() {
    const fileInput = document.getElementById("imageInput");
    const statusText = document.getElementById("status");

    if (fileInput.files.length === 0) {
        statusText.innerText = "Veuillez sélectionner une image.";
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("image", file);

    // Afficher l'image originale
    const originalImage = document.getElementById("originalImage");
    originalImage.src = URL.createObjectURL(file);
    originalImage.style.display = "block";

    statusText.innerText = "Traitement en cours...";
    
    fetch("http://127.0.0.1:5000/enhance", {
        method: "POST",
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        const enhancedImage = document.getElementById("enhancedImage");
        enhancedImage.src = URL.createObjectURL(blob);
        enhancedImage.style.display = "block";
        statusText.innerText = "Traitement terminé ✅";
    })
    .catch(error => {
        console.error("Erreur :", error);
        statusText.innerText = "Erreur lors du traitement ❌";
    });
}

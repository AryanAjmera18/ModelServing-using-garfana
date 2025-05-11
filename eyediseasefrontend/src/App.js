import React, { useState } from "react";
import axios from "axios";
import { Loader2, UploadCloud, AlertTriangle } from "lucide-react";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      console.log("Selected file:", selected);
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setPrediction(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await axios.post("http://127.0.0.1:8000/predict", formData);
      setPrediction(res.data);
    } catch (err) {
      console.error(err);
      setError("Prediction failed. Please try again.");
    }
    setLoading(false);
  };

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center justify-center px-4 py-10">
      <h1 className="text-3xl font-semibold mb-6 tracking-tight">
        👁️ AI Eye Disease Detection
      </h1>

      <div className="w-full max-w-md bg-gray-900 border border-gray-700 rounded-2xl shadow-xl p-6">
        {preview ? (
          <img
            src={preview}
            alt="Preview"
            className="w-64 h-64 object-cover rounded-xl shadow-md border border-gray-700 mb-4 mx-auto"
          />
        ) : (
          <div className="w-64 h-64 flex items-center justify-center border-2 border-dashed border-gray-700 rounded-xl mb-4 mx-auto">
            <UploadCloud className="text-gray-500 w-12 h-12" />
          </div>
        )}

        <div className="flex flex-col gap-2 mb-4">
          <input
            id="file-upload"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <button
            onClick={() => document.getElementById("file-upload").click()}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-xl transition"
          >
            Choose Image
          </button>
        </div>

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-xl transition flex items-center justify-center"
        >
          {loading && <Loader2 className="animate-spin mr-2" />}
          {loading ? "Analyzing..." : "Analyze Image"}
        </button>

        {prediction && (
          <div className="mt-6 text-center">
            <h2 className="text-lg font-medium mb-1">Prediction:</h2>
            <p className="text-xl font-semibold text-green-400">
              {prediction.predicted_class}
            </p>
            <p className="text-sm text-gray-400">
              Confidence: {(prediction.confidence * 100).toFixed(2)}%
            </p>
          </div>
        )}

        {error && (
          <div className="mt-6 text-red-500 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5" /> {error}
          </div>
        )}
      </div>
    </main>
  );
}

export default App;

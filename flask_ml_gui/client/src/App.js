import { useState } from 'react';
import axios from 'axios';

function App() {
  const [formData, setFormData] = useState({
    fixedAcidity: 0,
    volatileAcidity: 0,
    citricAcid: 0,
    residualSugar: 0,
    chlorides: 0,
    freeSulfurDioxide: 0,
    totalSulfurDioxide: 0,
    pH: 0,
    sulphates: 0,
    alcohol: 0,
  });
  const [prediction, setPrediction] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:5000/predict', formData);
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('Error making prediction:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 py-6 flex flex-col justify-center sm:py-12">
      <div className="relative py-3 sm:max-w-xl sm:mx-auto">
        <div className="relative px-4 py-10 bg-white shadow-lg sm:rounded-3xl sm:p-20">
          <div className="mb-8 text-center">
            <h1 className="text-3xl font-bold text-gray-900">Wine Quality Prediction</h1>
            <p className="mt-2 text-gray-600">Enter the wine physicochemical properties to predict its quality.</p>
            <p className="mt-2 text-gray-600">The quality is predicted on a scale of low to medium to high.</p>
          </div>
          <form onSubmit={handleSubmit} className="space-y-4">
            {Object.keys(formData).map((key) => (
              <div key={key}>
                <label className="block text-sm font-medium text-gray-700">
                  {key}
                </label>
                <input
                  type="number"
                  step="0.01"
                  className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
                  value={formData[key]}
                  onChange={(e) => setFormData({
                    ...formData,
                    [key]: e.target.value
                  })}
                />
              </div>
            ))}
            <button
              type="submit"
              className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700"
            >
              Predict
            </button>
          </form>
          {prediction && (
            <div className="mt-4 text-center">
              <h2 className="text-xl font-bold">Prediction: {prediction} quality wine</h2>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;

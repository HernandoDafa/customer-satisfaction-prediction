document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('formPrediksi').addEventListener('submit', function(e) {
        e.preventDefault();

        // Ambil nilai dari formulir
        let umur = document.getElementById('umur').value;
        let jenisKelamin = document.getElementById('jenisKelamin').value;
        let kota = document.getElementById('kota').value;
        // Ambil nilai dari inputan lainnya sesuai kebutuhan
        
        // Buat objek data
        let data = {
            umur: umur,
            jenisKelamin: jenisKelamin,
            kota: kota,
            // Tambahkan nilai inputan lainnya ke dalam objek data
        };

        // Kirim data ke backend menggunakan fetch
        fetch('URL_BACKEND_ANDA', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            // Tampilkan hasil prediksi di halaman
            document.getElementById('hasil').innerHTML = `
                <p>Prediksi Kepuasan Pelanggan: ${result.prediksi}</p>
                <p>Skor Probabilitas: ${result.probabilitas}</p>
                <p>Model yang Digunakan: ${result.model}</p>
            `;

            // Tampilkan metrik model
            document.getElementById('metrikModel').innerHTML = `
                <p>Akurasi: ${result.akurasi}</p>
                <p>Presisi: ${result.presisi}</p>
                <p>Recall: ${result.recall}</p>
                <p>F1-Score: ${result.f1}</p>
            `;

            // Tampilkan pentingnya fitur jika ada
            if (result.featureImportance) {
                let featureHtml = '<ul>';
                result.featureImportance.forEach(feature => {
                    featureHtml += `<li>${feature.nama}: ${feature.nilai}</li>`;
                });
                featureHtml += '</ul>';
                document.getElementById('featureImportance').innerHTML = featureHtml;
            }

            // Tampilkan confusion matrix jika ada
            if (result.confusionMatrix) {
                let matrixHtml = '<table>';
                result.confusionMatrix.forEach(row => {
                    matrixHtml += '<tr>';
                    row.forEach(cell => {
                        matrixHtml += `<td>${cell}</td>`;
                    });
                    matrixHtml += '</tr>';
                });
                matrixHtml += '</table>';
                document.getElementById('confusionMatrix').innerHTML = matrixHtml;
            }

            // Tampilkan grafik perbandingan model jika ada
            if (result.modelComparison) {
                let comparisonHtml = '<ul>';
                result.modelComparison.forEach(model => {
                    comparisonHtml += `<li>${model.nama}: ${model.nilai}</li>`;
                });
                comparisonHtml += '</ul>';
                document.getElementById('perbandinganModel').innerHTML = comparisonHtml;
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});

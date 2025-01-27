document.getElementById('prediction-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = new FormData(this);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        } 
        return response.json();
    })
    .then(data => {
        const resultDiv = document.getElementById('prediction-result');
        resultDiv.innerHTML = `
        <div class="result-table">
            <h2>Customer Risk</h2>
            <table>
                <tr>
                    <th>Customer ID</th>
                    <th>Risk Label</th>

                </tr>
                <tr>
                    <td>${data.customer_id}</td>
                    <td>${data.predicted_risk}</td>
                </tr>
            </table>
        </div>`;
    })
    .catch(error => {
        const resultDiv = document.getElementById('prediction-result');
        resultDiv.innerHTML = `<h2>Error: Unable to fetch prediction.</h2><p>${error.message}</p>`;
        console.error('Error:', error);
    });
});





/*document.getElementById('predictBtn').addEventListener('click', async function () {
    const TransactionId = document.getElementById('TransactionId').value;
    const CustomerID = document.getElementById('CustomerId').value;
    const promo = document.getElementById('promo').value;

    if (!TransactionId || !CustomerID || !promo) {
        alert('Please fill in all fields');
        return;
    }

    const data = {
        store_id: parseInt(storeId),
        day_of_week: parseInt(dayOfWeek),
        promo: parseInt(promo)
    };

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            const result = await response.json();
            document.getElementById('rfResult').textContent = result.prediction;
        } else {
            const error = await response.json();
            alert('Error: ' + error.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
});*/
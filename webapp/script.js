document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const resultDisplay = document.getElementById('resultDisplay');
    const dtiVal = document.getElementById('val-dti');
    const ltvVal = document.getElementById('val-ltv');
    const defaultProbVal = document.getElementById('val-default-prob');
    const modelVal = document.getElementById('val-model');

    // Loading state
    resultDisplay.className = 'result-display';
    resultDisplay.innerHTML = '<div class="loader"></div><p style="margin-top:15px;">Analyzing financial profile...</p>';

    // Collect form values
    const payload = {
        age: document.getElementById('age').value,
        employment_status: document.getElementById('employment_status').value,
        prev_default: document.getElementById('prev_default').value,
        income: document.getElementById('income').value,
        savings: document.getElementById('savings').value,
        loan_amount: document.getElementById('loan_amount').value,
        collateral: document.getElementById('collateral').value,
        loan_duration: document.getElementById('loan_duration').value,
        guarantor_count: document.getElementById('guarantor_count').value,
        membership_years: document.getElementById('membership_years').value,
        previous_loans_count: document.getElementById('previous_loans_count').value,
        interest_rate: document.getElementById('interest_rate').value,
        model_choice: document.getElementById('model_choice').value
    };

    // Compute local metrics for display
    const income = parseFloat(payload.income);
    const loan = parseFloat(payload.loan_amount);
    const duration = parseFloat(payload.loan_duration);
    const collateral = parseFloat(payload.collateral);

    const dti = ((loan / duration) / (income + 1)) * 100;
    const ltv = (loan / (collateral + 1)) * 100;

    dtiVal.innerText = dti.toFixed(1) + '%';
    ltvVal.innerText = ltv.toFixed(1) + '%';
    dtiVal.style.color = dti < 40 ? '#10b981' : '#ef4444';
    ltvVal.style.color = ltv < 80 ? '#10b981' : '#ef4444';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        // No fake delay — render immediately
        if (response.status === 400) {
            resultDisplay.className = 'result-display result-denied';
            resultDisplay.innerHTML = `<h3>Data Error</h3><p>${data.message}</p>`;
            return;
        }

        if (data.is_approved) {
            resultDisplay.className = 'result-display result-approved';
            resultDisplay.innerHTML = `<h3>${data.status}</h3><p>Confidence: ${data.confidence}%</p>`;
        } else {
            resultDisplay.className = 'result-display result-denied';
            resultDisplay.innerHTML = `<h3>${data.status}</h3><p>Confidence: ${data.confidence}%</p>`;
        }

        // Update metrics from API response
        if (data.default_probability !== undefined) {
            defaultProbVal.innerText = data.default_probability.toFixed(1) + '%';
            defaultProbVal.style.color = data.default_probability < 50 ? '#10b981' : '#ef4444';
        }
        if (data.model_used) {
            modelVal.innerText = data.model_used;
        }
        if (data.dti !== undefined) {
            dtiVal.innerText = data.dti.toFixed(1) + '%';
        }
        if (data.ltv !== undefined) {
            ltvVal.innerText = data.ltv.toFixed(1) + '%';
        }

        // SHAP Decision Drivers
        if (data.explanations && data.explanations.length > 0) {
            let explainHtml = '<div style="margin-top:30px; text-align:left; font-size:1.1rem; width: 100%;">';
            explainHtml += '<h4 style="margin-bottom:15px; border-bottom: 3px solid #000; padding-bottom: 5px; text-transform: uppercase; font-weight: 700;">AI Decision Drivers</h4>';
            explainHtml += '<ul style="list-style-type:none; padding:0;">';
            data.explanations.forEach(factor => {
                const isDanger = factor.direction.includes('increase');
                const color = isDanger ? '#ef4444' : '#10b981';
                const arrow = isDanger ? '▲' : '▼';
                explainHtml += `<li style="margin-bottom:10px; padding:10px; border:2px solid #000; background:#fff;">
                    <span style="color:${color}; font-weight:bold;">${arrow}</span>
                    <strong> ${factor.feature}</strong>
                    <span style="float:right; color:${color}; font-weight:bold;">${factor.direction}</span>
                </li>`;
            });
            explainHtml += '</ul></div>';
            resultDisplay.innerHTML += explainHtml;
        }

        // Policy guardrail violations
        if (data.guardrail_violations && data.guardrail_violations.length > 0) {
            let ruleHtml = '<div style="margin-top:20px; text-align:left; width:100%;">';
            ruleHtml += '<h4 style="margin-bottom:10px; text-transform:uppercase; font-weight:700;">Policy Rules Triggered</h4>';
            ruleHtml += '<ul style="padding-left:20px;">';
            data.guardrail_violations.forEach(rule => {
                ruleHtml += `<li style="margin-bottom:6px; color:#ef4444;">${rule}</li>`;
            });
            ruleHtml += '</ul></div>';
            resultDisplay.innerHTML += ruleHtml;
        }

    } catch (error) {
        resultDisplay.className = 'result-display result-denied';
        resultDisplay.innerHTML = `<h3>Connection Error</h3>
            <p>Could not reach the prediction API at localhost:5000.</p>
            <p style="font-size:0.9rem; margin-top:10px;">Make sure the Flask server is running: <code>python api.py</code></p>`;
    }
});
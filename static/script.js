document.addEventListener('DOMContentLoaded', function() {
    setupTimeSlider();
});

function setupTimeSlider() {
    const timeSlider = document.getElementById('currentTime');
    const timeDisplay = document.getElementById('timeDisplay');
    
    timeSlider.addEventListener('input', function() {
        const hour = parseInt(this.value);
        const timeStr = String(hour).padStart(2, '0') + ':00';
        timeDisplay.textContent = timeStr;
    });
    
    const now = new Date();
    timeSlider.value = now.getHours();
    timeSlider.dispatchEvent(new Event('input'));
}

async function analyzeCustomer() {
    const budgetLevel = document.getElementById('budgetLevel').value;
    const foodType = document.getElementById('foodType').value;
    const occasion = document.getElementById('occasion').value;
    const customerType = document.getElementById('customerType').value;
    const deliveryPref = document.getElementById('deliveryPref').value;
    const paymentMethod = document.getElementById('paymentMethod').value;
    const time = parseInt(document.getElementById('currentTime').value);

    if (!budgetLevel || !foodType || !occasion || !customerType || !deliveryPref) {
        alert('Please fill in all required fields');
        return;
    }

    const criteria = {
        time: time,
        budget_level: budgetLevel,
        food_type: foodType,
        occasion: occasion,
        customer_type: customerType,
        delivery_preference: deliveryPref,
        payment_method: paymentMethod
    };

    try {
        const response = await fetch('/api/identify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(criteria)
        });

        const result = await response.json();
        displayAnalysisResult(result);
    } catch (error) {
        console.error('Error analyzing customer:', error);
        document.getElementById('analyzerResult').innerHTML = 
            '<div class="error-message"><i class="fas fa-exclamation-circle"></i> Error analyzing customer. Please try again.</div>';
    }
}

function displayAnalysisResult(result) {
    const container = document.getElementById('analyzerResult');

    if (result.confidence === 0) {
        container.innerHTML = `<div class="empty-state"><p>${result.message}</p></div>`;
        return;
    }

    const iconMap = {
        'street_food': '[SF]',
        'fast_casual': '[FC]',
        'fine_dining': '[FD]',
        'cloud_kitchen': '[CK]',
        'regional_specialty': '[RS]',
        'catering_events': '[CE]',
        'health_organic': '[HO]'
    };

    const characteristics = result.characteristics.map(c => `<span class="tag">${c}</span>`).join('');
    const reasons = result.detection_reasons.map(r => `<li>${r}</li>`).join('');
    const products = result.products.map(p => `<span class="tag">${p}</span>`).join('');
    const cuisines = (result.cuisine_types || []).map(c => `<span class="tag">${c}</span>`).join('');
    const payments = (result.payment_methods || []).map(p => `<span class="tag">${p}</span>`).join('');
    const strategies = result.marketing_strategies.map(s => `<span class="tag">${s}</span>`).join('');

    const confidenceColor = result.confidence >= 70 ? '#10b981' : result.confidence >= 40 ? '#f59e0b' : '#ef4444';

    container.innerHTML = `
        <div class="result-content">
            <div class="result-header-card">
                <div class="persona-icon">${iconMap[result.persona_key] || '[?]'}</div>
                <div class="result-title-block">
                    <h3 class="persona-name">${result.persona_name}</h3>
                    <div class="confidence-display">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${result.confidence}%; background-color: ${confidenceColor};"></div>
                        </div>
                        <span class="confidence-text">Confidence: ${Math.round(result.confidence)}%</span>
                    </div>
                </div>
            </div>

            <div class="result-section">
                <h4><i class="fas fa-lightbulb"></i> Why This Profile?</h4>
                <ul class="reasons-list">${reasons}</ul>
            </div>

            <div class="result-section">
                <h4><i class="fas fa-user-check"></i> Key Characteristics</h4>
                <div class="tag-group">${characteristics}</div>
            </div>

            <div class="result-grid">
                <div class="result-box">
                    <h5><i class="fas fa-clock"></i> Peak Hours</h5>
                    <div class="tag-group">
                        ${result.peak_times.map(t => `<span class="tag">${t}</span>`).join('')}
                    </div>
                </div>

                <div class="result-box">
                    <h5><i class="fas fa-wallet"></i> Budget Range</h5>
                    <p class="highlight-text">${result.avg_spending}</p>
                </div>

                <div class="result-box">
                    <h5><i class="fas fa-utensils"></i> Cuisines</h5>
                    <div class="tag-group">${cuisines}</div>
                </div>

                <div class="result-box">
                    <h5><i class="fas fa-credit-card"></i> Payment Methods</h5>
                    <div class="tag-group">${payments}</div>
                </div>
            </div>

            <div class="result-section">
                <h4><i class="fas fa-shopping-bag"></i> Menu Recommendations</h4>
                <div class="tag-group">${products}</div>
            </div>

            <div class="result-section">
                <h4><i class="fas fa-chart-line"></i> Pricing Strategy</h4>
                <div class="strategy-box">
                    <p>${result.price_strategy}</p>
                </div>
            </div>

            <div class="result-section">
                <h4><i class="fas fa-megaphone"></i> Marketing Channels</h4>
                <div class="tag-group">${strategies}</div>
            </div>

            ${result.all_predictions ? `
            <div class="result-section">
                <h4><i class="fas fa-brain"></i> Model Confidence Breakdown</h4>
                <div class="tag-group">
                    ${Object.entries(result.all_predictions).map(([persona, prob]) => 
                        `<span class="tag" style="opacity: ${Math.max(0.5, prob)};">${persona}: ${(prob*100).toFixed(1)}%</span>`
                    ).join('')}
                </div>
            </div>
            ` : ''}
        </div>
    `;
}

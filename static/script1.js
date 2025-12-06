let userLocation = { latitude: 28.7041, longitude: 77.1025 };

const cityRegionMap = {
    'Amaravati': 'South India',
    'Itanagar': 'North East India',
    'Dispur': 'North East India',
    'Patna': 'East India',
    'Raipur': 'Central India',
    'Panaji': 'West India',
    'Gandhinagar': 'West India',
    'Chandigarh': 'North India',
    'Shimla': 'North India',
    'Ranchi': 'East India',
    'Bengaluru': 'South India',
    'Thiruvananthapuram': 'South India',
    'Bhopal': 'Central India',
    'Mumbai': 'West India',
    'Imphal': 'North East India',
    'Shillong': 'North East India',
    'Aizawl': 'North East India',
    'Kohima': 'North East India',
    'Bhubaneswar': 'East India',
    'Jaipur': 'North India',
    'Gangtok': 'North East India',
    'Chennai': 'South India',
    'Hyderabad': 'South India',
    'Agartala': 'North East India',
    'Lucknow': 'North India',
    'Dehradun': 'North India',
    'Kolkata': 'East India',
    'New Delhi': 'North India'
};

function updateLocationFromCity() {
    const citySelect = document.getElementById('city');
    const selectedOption = citySelect.options[citySelect.selectedIndex];
    
    const lat = parseFloat(selectedOption.dataset.lat);
    const lon = parseFloat(selectedOption.dataset.lon);
    const cityName = selectedOption.value;
    
    userLocation = { latitude: lat, longitude: lon };
    
    if (cityRegionMap[cityName]) {
        document.getElementById('region').value = cityRegionMap[cityName];
    }
    
    fetchWeatherData();
}

function getForecast() {
    const city = document.getElementById('city').options[document.getElementById('city').selectedIndex].text;
    const region = document.getElementById('region').value;
    const temperature = parseFloat(document.getElementById('temperature').value);
    const humidity = parseFloat(document.getElementById('humidity').value);
    const weather_condition = document.getElementById('weather').value;

    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');

    loading.style.display = 'block';
    results.style.display = 'none';
    error.style.display = 'none';

    fetch('/api/forecast', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            city,
            region,
            temperature,
            humidity,
            weather_condition,
            latitude: userLocation.latitude,
            longitude: userLocation.longitude
        })
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        displayResults(data);
        results.style.display = 'block';
        getForecast24h(city, region);
    })
    .catch(err => {
        loading.style.display = 'none';
        error.style.display = 'block';
        document.getElementById('error-message').textContent = 'Error: ' + err.message;
    });
}

function fetchUserLocation() {
    fetch('/api/location')
        .then(response => response.json())
        .then(data => {
            userLocation = { latitude: data.latitude, longitude: data.longitude };
            document.getElementById('city').value = data.city;
            fetchWeatherData();
        })
        .catch(err => console.warn('Location fetch failed, using defaults:', err));
}

function fetchWeatherData() {
    const params = new URLSearchParams({
        latitude: userLocation.latitude,
        longitude: userLocation.longitude
    });
    
    fetch(`/api/weather?${params}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById('temperature').value = data.temperature;
            document.getElementById('humidity').value = data.humidity;
            
            const weatherMap = {
                'clear': 'clear',
                'partly_cloudy': 'clear',
                'overcast': 'clear',
                'foggy': 'clear',
                'rainy': 'rainy',
                'snow': 'cold'
            };
            
            const mappedWeather = weatherMap[data.weather_condition] || 'clear';
            document.getElementById('weather').value = mappedWeather;
        })
        .catch(err => console.warn('Weather fetch failed, using defaults:', err));
}

function displayResults(data) {
    document.getElementById('result-city').textContent = data.city;
    document.getElementById('result-region').textContent = data.region;
    document.getElementById('result-season').textContent = capitalizeFirst(data.season);
    document.getElementById('result-weather').textContent = capitalizeFirst(data.weather);

    const forecast = data.forecast;

    document.getElementById('weather-foods').innerHTML = 
        forecast.weather_based.foods.map(f => `<span class="food-tag">${formatFoodName(f)}</span>`).join('');

    document.getElementById('local-foods').innerHTML = 
        forecast.local_taste.foods.map(f => `<span class="food-tag">${formatFoodName(f)}</span>`).join('');

    document.getElementById('seasonal-foods').innerHTML = 
        forecast.seasonal_demand.foods.map(f => `<span class="food-tag">${formatFoodName(f)}</span>`).join('');

    const timeDemandHtml = Object.entries(forecast.time_of_day.demand_pattern)
        .map(([food, score]) => `<div class="demand-item">${formatFoodName(food)} <span class="demand-value">${(score * 100).toFixed(0)}%</span></div>`)
        .join('');
    document.getElementById('time-demand').innerHTML = timeDemandHtml;

    const dayDemandHtml = Object.entries(forecast.day_pattern.demand_pattern)
        .map(([food, score]) => `<div class="demand-item">${formatFoodName(food)} <span class="demand-value">${(score * 100).toFixed(0)}%</span></div>`)
        .join('');
    document.getElementById('day-demand').innerHTML = dayDemandHtml;

    const combinedDemandHtml = forecast.combined_demand
        .slice(0, 10)
        .map((item, idx) => `
            <div class="ranking-item">
                <div>
                    <div class="ranking-item-name">#${idx + 1} ${formatFoodName(item.food)}</div>
                    <div class="ranking-item-sources">Sources: ${item.sources.join(', ')}</div>
                </div>
                <div class="ranking-item-score">${(item.demand_score * 100).toFixed(1)}%</div>
            </div>
        `)
        .join('');
    document.getElementById('combined-demand').innerHTML = combinedDemandHtml;
}

function getForecast24h(city, region) {
    fetch('/api/detailed-forecast', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            city,
            region
        })
    })
    .then(response => response.json())
    .then(data => {
        displayForecast24h(data);
    })
    .catch(err => console.error('Error fetching 24h forecast:', err));
}

function displayForecast24h(data) {
    const forecast24h = document.getElementById('forecast-24h');
    const timelineHtml = data.forecast_24h
        .filter((_, idx) => idx % 2 === 0)
        .map(slot => `
            <div class="time-slot">
                <div class="time-slot-hour">${String(slot.hour).padStart(2, '0')}:00</div>
                <div class="time-slot-foods">
                    ${slot.foods.slice(0, 2).map(f => `<div>${formatFoodName(f)}</div>`).join('')}
                </div>
            </div>
        `)
        .join('');
    
    document.getElementById('forecast-timeline').innerHTML = timelineHtml;
    forecast24h.style.display = 'block';
}

function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function formatFoodName(name) {
    return name
        .split('_')
        .map(word => capitalizeFirst(word))
        .join(' ');
}

document.addEventListener('DOMContentLoaded', function() {
    const citySelect = document.getElementById('city');
    const temperature = document.getElementById('temperature');
    
    citySelect.addEventListener('change', updateLocationFromCity);
    
    const now = new Date();
    const month = now.getMonth() + 1;
    const hour = now.getHours();
    
    if (month >= 11 || month <= 2) {
        temperature.value = '15';
    } else if (month >= 5 && month <= 8) {
        temperature.value = '35';
    } else {
        temperature.value = '25';
    }

    updateLocationFromCity();

    document.querySelector('section').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            getForecast();
        }
    });
});

from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy import stats
import plotly
import plotly.graph_objects as go
import json
from utils import plot_distribution, calculate_binomial_probabilities

app = Flask(__name__)

# Dictionary of available distributions and their properties
DISTRIBUTIONS = {
    'continuous': {
        'normal': {
            'name': 'Normal Distribution',
            'parameters': ['mean', 'std'],
            'story': 'The normal distribution is the most important probability distribution in statistics. It describes many natural phenomena and is characterized by its bell-shaped curve.',
            'pdf': lambda x, mean, std: stats.norm.pdf(x, mean, std),
            'cdf': lambda x, mean, std: stats.norm.cdf(x, mean, std),
            'mgf': 'M(t) = exp(μt + σ²t²/2)'
        },
        'exponential': {
            'name': 'Exponential Distribution',
            'parameters': ['lambda'],
            'story': 'The exponential distribution describes the time between events in a Poisson process. It is memoryless and is often used to model waiting times.',
            'pdf': lambda x, lambda_: stats.expon.pdf(x, scale=1/lambda_),
            'cdf': lambda x, lambda_: stats.expon.cdf(x, scale=1/lambda_),
            'mgf': 'M(t) = λ/(λ-t) for t < λ'
        }
    },
    'discrete': {
        'poisson': {
            'name': 'Poisson Distribution',
            'parameters': ['lambda'],
            'story': 'The Poisson distribution describes the number of events occurring in a fixed time interval, given a constant average rate of occurrence.',
            'pdf': lambda x, lambda_: stats.poisson.pmf(x, lambda_),
            'cdf': lambda x, lambda_: stats.poisson.cdf(x, lambda_),
            'mgf': 'M(t) = exp(λ(e^t - 1))'
        },
        'binomial': {
            'name': 'Binomial Distribution',
            'parameters': ['n', 'p'],
            'story': 'The binomial distribution describes the number of successes in n independent Bernoulli trials, each with probability p of success.',
            'pdf': lambda x, n, p: stats.binom.pmf(x, n, p),
            'cdf': lambda x, n, p: stats.binom.cdf(x, n, p),
            'mgf': 'M(t) = (pe^t + (1-p))^n'
        }
    }
}

@app.route('/')
def index():
    # Render the home page with the list of distributions
    return render_template('index.html', distributions=DISTRIBUTIONS)

@app.route('/distribution/<dist_type>/<dist_name>')
def distribution(dist_type, dist_name):
    # Render the distribution detail page
    if dist_type in DISTRIBUTIONS and dist_name in DISTRIBUTIONS[dist_type]:
        dist_info = DISTRIBUTIONS[dist_type][dist_name]
        return render_template('distribution.html', 
                             dist_type=dist_type,
                             dist_name=dist_name,
                             dist_info=dist_info,
                             distributions=DISTRIBUTIONS)
    return "Distribution not found", 404

@app.route('/calculate', methods=['POST'])
def calculate():
    # Handle AJAX calculation requests for PDF/CDF and graph data
    data = request.get_json()
    dist_type = data.get('dist_type')
    dist_name = data.get('dist_name')
    calc_type = data.get('calc_type')
    x = float(data.get('x', 0))
    
    # Initialize parameters and statistics
    params_dict = {}
    mean_val, std_val = 0, 0
    dist = None
    x_range_plot = (0, 1)  # Default minimal range
    
    # Get parameters based on distribution type and determine plot range
    if dist_name == 'normal':
        mean = float(data.get('mean', 0))
        std = float(data.get('std', 1))
        if std <= 0:
            return jsonify({'error': 'Standard deviation must be positive'}), 400
        params_dict = {'mu': mean, 'sigma': std}
        dist = stats.norm(loc=mean, scale=std)
        x_range_plot = (mean - 4 * std, mean + 4 * std)  # Cover 4 standard deviations
        mean_val = mean
        std_val = std
    elif dist_name == 'exponential':
        lambda_ = float(data.get('lambda', 1))
        if lambda_ <= 0:
            return jsonify({'error': 'Lambda must be positive'}), 400
        params_dict = {'lambda': lambda_}
        dist = stats.expon(scale=1/lambda_)
        x_range_plot = (0, 5 / lambda_)  # 5 times the mean covers most of the distribution
        mean_val = 1/lambda_
        std_val = 1/lambda_
    elif dist_name == 'binomial':
        n = int(data.get('n', 10))
        p = float(data.get('p', 0.5))
        if n <= 0 or not (0 <= p <= 1):
            return jsonify({'error': 'N must be positive and P between 0 and 1'}), 400
        params_dict = {'n': n, 'p': p}
        dist = stats.binom(n=n, p=p)
        x_range_plot = (0, n)  # Covers all possible outcomes from 0 to n
        mean_val = n * p
        std_val = np.sqrt(n * p * (1-p))
    elif dist_name == 'poisson':
        lambda_ = float(data.get('lambda', 1))
        if lambda_ <= 0:
            return jsonify({'error': 'Lambda must be positive'}), 400
        params_dict = {'lambda': lambda_}
        dist = stats.poisson(mu=lambda_)
        max_x = int(lambda_ + 4 * np.sqrt(lambda_))
        if max_x < 10:
            max_x = 10
        x_range_plot = (0, max_x)
        mean_val = lambda_
        std_val = np.sqrt(lambda_)
    else:
        return jsonify({'error': 'Unknown distribution'}), 400
    
    # For discrete distributions, x must be an integer for calculations
    x_calc = int(round(x)) if dist_type == 'discrete' else x
    
    # Generate points for the plot
    if dist_type == 'continuous':
        x_vals = np.linspace(x_range_plot[0], x_range_plot[1], 500)
    else:
        x_vals = np.arange(x_range_plot[0], x_range_plot[1] + 1)
    
    if calc_type == 'pdf':
        y_vals = dist.pdf(x_vals) if dist_type == 'continuous' else dist.pmf(x_vals)
    else:
        y_vals = dist.cdf(x_vals)
    
    # Create points array for D3.js
    points = [{'x': float(x), 'y': float(y)} for x, y in zip(x_vals, y_vals)]
    
    # Calculate probabilities
    if dist_type == 'discrete':
        prob_exact = float(dist.pmf(x_calc))
        prob_less = float(dist.cdf(x_calc - 1))
        prob_less_equal = float(dist.cdf(x_calc))
        prob_greater = 1 - prob_less_equal
        prob_greater_equal = 1 - prob_less
    else:  # continuous
        prob_exact = 0.0  # P(X = x) is always 0 for continuous distributions
        prob_less = float(dist.cdf(x_calc))
        prob_less_equal = float(dist.cdf(x_calc))
        prob_greater = 1 - prob_less_equal
        prob_greater_equal = 1 - prob_less
    
    return jsonify({
        'prob_exact': prob_exact,
        'prob_less': prob_less,
        'prob_less_equal': prob_less_equal,
        'prob_greater': prob_greater,
        'prob_greater_equal': prob_greater_equal,
        'mean': mean_val,
        'std': std_val,
        'x_range': [float(x_range_plot[0]), float(x_range_plot[1])],
        'y_values': [float(y) for y in y_vals],
        'points': points
    })

@app.route('/calculate_binomial', methods=['POST'])
def calculate_binomial():
    # (Optional) Endpoint for detailed binomial calculations
    data = request.get_json()
    n = int(data.get('n'))
    p = float(data.get('p'))
    x = int(data.get('x'))
    results = calculate_binomial_probabilities(n, p, x)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
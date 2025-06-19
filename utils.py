import numpy as np
import plotly.graph_objects as go
from scipy import stats

# Utility to generate plot data for a probability distribution
# Returns points for D3.js (or other) plotting
# Handles both continuous and discrete distributions

def plot_distribution(dist_name, params, x_range, mode='pdf', shade_range=None):
    """
    Create a Plotly figure for a probability distribution.
    Args:
        dist_name (str): Name of the distribution ('normal', 'exponential', 'binomial', 'poisson')
        params (dict): Distribution parameters
        x_range (tuple): Range for x-axis (min, max)
        mode (str): 'pdf' or 'cdf'
        shade_range (tuple): Range to shade (min, max)
    Returns:
        plotly.graph_objects.Figure: The plot figure
    """
    # Generate x values
    if dist_name in ['normal', 'exponential']:
        x = np.linspace(x_range[0], x_range[1], 500)
    else:  # discrete distributions
        x = np.arange(x_range[0], x_range[1] + 1)
    
    # Get distribution function
    if dist_name == 'normal':
        dist = stats.norm(loc=params['mu'], scale=params['sigma'])
    elif dist_name == 'exponential':
        dist = stats.expon(scale=1/params['lambda'])
    elif dist_name == 'binomial':
        dist = stats.binom(n=params['n'], p=params['p'])
    elif dist_name == 'poisson':
        dist = stats.poisson(mu=params['lambda'])
    
    # Calculate y values
    if dist_name in ['normal', 'exponential']:
        y = dist.pdf(x) if mode == 'pdf' else dist.cdf(x)
    else:  # discrete distributions
        y = dist.pmf(x) if mode == 'pdf' else dist.cdf(x)
    
    # Create points for D3.js
    points = [{'x': float(xi), 'y': float(yi)} for xi, yi in zip(x, y)]
    return points

# Utility to calculate all binomial probabilities and statistics

def calculate_binomial_probabilities(n, p, x):
    """
    Calculate all binomial probabilities for given parameters.
    Args:
        n (int): Number of trials
        p (float): Probability of success
        x (int): Number of successes
    Returns:
        dict: Dictionary containing all probabilities and statistics
    """
    dist = stats.binom(n=n, p=p)
    # Calculate individual probabilities
    prob_exact = dist.pmf(x)
    prob_less = dist.cdf(x-1)
    prob_less_equal = dist.cdf(x)
    prob_greater = 1 - dist.cdf(x)
    prob_greater_equal = 1 - dist.cdf(x-1)
    # Calculate mean and standard deviation
    mean = n * p
    std = np.sqrt(n * p * (1-p))
    # Calculate probability distribution table
    x_vals = np.arange(n+1)
    pmf_vals = dist.pmf(x_vals)
    cdf_vals = dist.cdf(x_vals)
    distribution_table = [
        {'x': int(x), 'pmf': f'{p:.4f}', 'cdf': f'{c:.4f}'}
        for x, p, c in zip(x_vals, pmf_vals, cdf_vals)
    ]
    return {
        'prob_exact': prob_exact,
        'prob_less': prob_less,
        'prob_less_equal': prob_less_equal,
        'prob_greater': prob_greater,
        'prob_greater_equal': prob_greater_equal,
        'mean': mean,
        'std': std,
        'distribution_table': distribution_table
    }

def plot_distribution(dist_name, params, x_range, calc_type='pdf', x_value=None, prob_type='less'):
    """
    Create a Plotly figure for a probability distribution with proper highlighting.
    
    Args:
        dist_name (str): Name of the distribution ('normal', 'exponential', 'binomial', 'poisson')
        params (dict): Distribution parameters
        x_range (tuple): Range for x-axis (min, max)
        calc_type (str): 'pdf' or 'cdf'
        x_value (float/int, optional): Specific x value for probability calculation
        prob_type (str): Type of probability calculation ('less', 'less_equal', 'greater', 'greater_equal')
    
    Returns:
        plotly.graph_objects.Figure: The plot figure
    """
    # Determine if distribution is discrete or continuous
    is_discrete = dist_name in ['binomial', 'poisson']
    
    # Generate x values
    if is_discrete:
        x = np.arange(int(x_range[0]), int(x_range[1]) + 1)
        if len(x) == 0:
            x = np.array([0])
    else:
        x = np.linspace(x_range[0], x_range[1], 500)
    
    # Get distribution object
    if dist_name == 'normal':
        dist = stats.norm(loc=params['mu'], scale=params['sigma'])
    elif dist_name == 'exponential':
        dist = stats.expon(scale=1/params['lambda'])
    elif dist_name == 'binomial':
        dist = stats.binom(n=params['n'], p=params['p'])
    elif dist_name == 'poisson':
        dist = stats.poisson(mu=params['lambda'])
    else:
        raise ValueError("Unsupported distribution name")
    
    # Calculate y values
    if calc_type == 'pdf':
        y = dist.pmf(x) if is_discrete else dist.pdf(x)
    else:
        y = dist.cdf(x)
    
    # Create figure
    fig = go.Figure()
    
    # For PDF plots with x_value, we'll highlight the relevant area
    if calc_type == 'pdf' and x_value is not None:
        if is_discrete:
            # Discrete case - highlight bars
            bar_colors = []
            highlight_indices = []
            
            if prob_type in ['less', 'less_equal']:
                compare_val = x_value if prob_type == 'less_equal' else x_value - 1
                highlight_indices = np.where(x <= compare_val)[0]
            else:  # greater or greater_equal
                compare_val = x_value if prob_type == 'greater_equal' else x_value + 1
                highlight_indices = np.where(x >= compare_val)[0]
            
            # Create color array
            bar_colors = ['red' if i in highlight_indices else 'royalblue' for i in range(len(x))]
            
            # Add bar trace
            fig.add_trace(go.Bar(
                x=x, y=y,
                marker_color=bar_colors,
                name=f'{dist_name.title()} PDF'
            ))
            
        else:
            # Continuous case - fill area under curve
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='royalblue', width=2),
                name=f'{dist_name.title()} PDF',
                fill=None
            ))
            
            # Determine fill range based on probability type
            if prob_type in ['less', 'less_equal']:
                fill_x = x[x <= x_value]
                fill_y = y[x <= x_value]
            else:  # greater or greater_equal
                fill_x = x[x >= x_value]
                fill_y = y[x >= x_value]
            
            # Add filled area
            fig.add_trace(go.Scatter(
                x=fill_x, y=fill_y,
                mode='lines',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.3)',
                name=f'P(X {get_inequality_symbol(prob_type)} {x_value:.2f})',
                showlegend=True
            ))
            
            # Add vertical line at x_value
            fig.add_vline(
                x=x_value,
                line=dict(color='red', width=2, dash='dash'),
                annotation_text=f'x = {x_value:.2f}'
            )
    
    else:
        # No highlighting - just basic plot
        if is_discrete:
            fig.add_trace(go.Bar(
                x=x, y=y,
                marker_color='royalblue',
                name=f'{dist_name.title()} {calc_type.upper()}'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                line=dict(color='royalblue', width=2),
                name=f'{dist_name.title()} {calc_type.upper()}'
            ))
    
    # Update layout
    title = f'{dist_name.title()} {calc_type.upper()}'
    if x_value is not None:
        title += f' - P(X {get_inequality_symbol(prob_type)} {x_value:.2f})'
    
    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title='Probability Density' if calc_type == 'pdf' else 'Cumulative Probability',
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Set y-axis range
    if len(y) > 0:
        y_max = np.max(y) * 1.1 if np.max(y) > 0 else 1
        fig.update_yaxes(range=[0, y_max])
    
    return fig

def get_inequality_symbol(prob_type):
    """Helper function to get inequality symbol for display"""
    symbols = {
        'less': '<',
        'less_equal': '≤',
        'greater': '>',
        'greater_equal': '≥'
    }
    return symbols.get(prob_type, '<')
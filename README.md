# Probability Distribution Visualizer

A Flask-based web application for visualizing and calculating various probability distributions. This tool is designed to help students and educators understand probability theory by providing interactive visualizations and calculations for both continuous and discrete probability distributions.

## 🌐 Live Demo

Visit the live application: [Probability Distribution Calculator](https://probability-dist-calculator.onrender.com)

> **Note**: This app is hosted on Render's free tier, which means it may take 30-60 seconds to spin up if it hasn't been accessed recently.

## Features

- Interactive visualization of PDFs and CDFs
- Support for both continuous (Normal, Exponential) and discrete (Binomial, Poisson) distributions
- Real-time calculations with parameter inputs
- Detailed information about each distribution including:
  - Distribution story/background
  - Moment Generating Function (MGF)
  - PDF and CDF calculations
- Modern, responsive UI with Bootstrap
- D3.js-powered interactive graphs

## Technologies Used

- **Backend:** Python, Flask, SciPy, NumPy
- **Frontend:** Bootstrap 5, D3.js, JavaScript (ES6+), HTML5

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ProbDist
   ```
2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application:**
   ```bash
   python app.py
   ```
5. **Open your web browser and navigate to** `http://localhost:5000`

## Usage

1. Select a distribution from the sidebar
2. Read about the distribution and its MGF
3. Use the calculator to:
   - Choose between PDF and CDF calculations
   - Enter the x value
   - Input the required parameters
   - View the result and interactive plot

## Contributing

Contributions are welcome! You can:

- Add more probability distributions
- Improve the UI/UX
- Add more features or calculations
- Report bugs or suggest improvements

To contribute:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🚀 Deployment

This application is deployed on [Render](https://render.com) using their free tier service.

### Deployment Status

[![Render](https://img.shields.io/badge/Render-%46E3B7.svg?style=for-the-badge&logo=render&logoColor=white)](https://probability-dist-calculator.onrender.com)

### Understanding the Deployment

- **Auto-Deploy**: The application automatically redeploys when changes are pushed to the main branch
- **Free Tier Characteristics**:
  - Service spins down after 15 minutes of inactivity
  - Takes 30-60 seconds to spin up on new requests
  - 750 free hours per month
  - 512 MB RAM and 0.1 CPU allocation

### Monitoring and Maintenance

To keep the application running smoothly:

1. **Monitor Application Logs**:

   - Check Render dashboard for errors and performance metrics
   - Review application logs for debugging

2. **Updates and Changes**:

   - Push changes to the GitHub repository
   - Render will automatically rebuild and deploy

3. **Best Practices**:
   - Regularly check the application status
   - Monitor resource usage in Render dashboard
   - Keep dependencies updated

# 🍽️ Zomato Restaurant Data Analysis

A comprehensive Python-based data analysis project for exploring and visualizing Zomato restaurant data. This tool provides insights into restaurant ratings, pricing, online ordering patterns, and customer voting behavior.


## 📊 Features

- **Data Cleaning & Preprocessing**: Automated data cleaning with handling of missing values and duplicates
- **Restaurant Type Analysis**: Distribution and popularity analysis across different restaurant categories
- **Top Performers Identification**: Find restaurants with highest votes and ratings
- **Online Order Patterns**: Analyze the relationship between online ordering and ratings
- **Rating Distribution**: Comprehensive rating analysis with statistical insights
- **Cost Analysis**: Examine pricing patterns and their correlation with ratings
- **Correlation Heatmaps**: Visual representation of relationships between categorical variables
- **Automated Reporting**: Generate comprehensive summary reports with key metrics

## 🎯 Sample Visualizations

The analysis generates 6+ high-quality visualizations including:
- Restaurant type distribution charts
- Top restaurants by votes
- Online order availability analysis
- Rating distributions and density plots
- Cost vs rating scatter plots
- Correlation heatmaps

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/zomato-restaurant-analysis.git
cd zomato-restaurant-analysis
```

2. **Create a virtual environment (recommended)**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Usage

1. **Place your dataset**
   - Add your `Zomato-data-.csv` file to the project root directory
   - Or update the `DATA_FILE` path in `zomato_analysis.py`
   

2. **Run the analysis**
```bash
python zomato_analysis.py
Or run the jupyter file - zomato.ipynb 
```

3. **View results**
   - All visualizations are saved in the `outputs/` folder
   - Summary report is generated as `outputs/summary_report.csv`
   - Charts are displayed in interactive windows during execution

## 📁 Project Structure

```
zomato-restaurant-analysis/
│
├── zomato_analysis.py          # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── LICENSE                      # License file
├── .gitignore                   # Git ignore rules
│
├── data/                        # Data directory (optional)
│   └── Zomato-data-.csv        # Your dataset
│
├── outputs/                     # Generated visualizations and reports
│   ├── restaurant_types_analysis.png
│   ├── top_restaurants.png
│   ├── online_order_analysis.png
│   ├── ratings_analysis.png
│   ├── cost_analysis.png
│   ├── correlation_heatmap.png
│   └── summary_report.csv
│
└── notebooks/                   # Jupyter notebooks (optional)
    └── exploratory_analysis.ipynb
```

## 📦 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🔧 Configuration

You can customize the analysis by modifying parameters in the `ZomatoAnalyzer` class:

```python
# Change output directory
analyzer = ZomatoAnalyzer('Zomato-data-.csv', output_dir='my_results')

# Adjust number of top restaurants
analyzer.find_top_restaurants(n=20)  # Default is 10
```

## 📈 Analysis Components

### 1. Data Cleaning
- Handles rating format conversion (e.g., "4.5/5" → 4.5)
- Removes duplicate entries
- Processes cost information
- Manages missing values

### 2. Exploratory Data Analysis
- Dataset overview and statistics
- Data type inspection
- Missing value analysis

### 3. Restaurant Type Analysis
- Distribution of restaurant categories
- Vote aggregation by type
- Visual comparisons

### 4. Top Performers
- Identifies highest-rated restaurants
- Analyzes most-voted establishments
- Generates ranking visualizations

### 5. Online Ordering Analysis
- Online order availability distribution
- Impact on ratings
- Comparison visualizations

### 6. Rating Analysis
- Distribution histograms
- Kernel density estimation
- Statistical summaries

### 7. Cost Analysis
- Price range distribution
- Cost vs rating correlation
- Trend analysis

### 8. Correlation Analysis
- Heatmaps for categorical relationships
- Cross-tabulation insights

## 🎨 Customization

### Modify Visualization Styles

```python
# Change color scheme
sns.set_palette("husl")

# Adjust figure sizes
plt.rcParams['figure.figsize'] = (12, 8)

# Change plot style
sns.set_style('darkgrid')
```

### Add Custom Analysis

You can extend the `ZomatoAnalyzer` class with your own methods:

```python
def analyze_cuisine_popularity(self):
    """Custom analysis for cuisine types."""
    # Your code here
    pass
```

## 📊 Sample Dataset Format

Your CSV should contain the following columns:
- `name`: Restaurant name
- `rate`: Rating (format: "X.X/5" or numeric)
- `votes`: Number of votes
- `online_order`: Yes/No
- `listed_in(type)`: Restaurant category
- `approx_cost(for two people)`: Approximate cost

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Known Issues

- Large datasets (>100,000 rows) may take longer to process
- Some visualizations may require adjusting figure sizes for datasets with many categories

## 📧 Contact

Your Name - smaoyabi@gmail.com

Project Link: [https://github.com/yourusername/zomato-restaurant-analysis](https://github.com/yourusername/zomato-restaurant-analysis)

## 🙏 Acknowledgments

- Zomato for the dataset
- Pandas, NumPy, Matplotlib, and Seaborn communities
- All contributors and supporters

## 📚 Additional Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

## 🔮 Future Enhancements

- [ ] Interactive dashboard with Plotly/Dash
- [ ] Machine learning models for rating prediction
- [ ] Geographic analysis with mapping
- [ ] Sentiment analysis on reviews
- [ ] API integration for real-time data
- [ ] Web application interface

---

**Made with ❤️ for data enthusiasts**

# Young Bull Picks - NBA PrizePicks Analysis Tool

A data-driven approach to analyzing NBA player prop betting lines for the **2024-2025 NBA season** using historical performance data, opponent matchups, and statistical modeling.

## Project Overview

This project applies data science techniques to evaluate PrizePicks betting lines for NBA players during the 2024-25 season. By combining multiple data sources and weighted factors, the model generates recommendations with confidence scores to help identify high-value betting opportunities.

---

## Data Science Process

### 1. Data Collection

The project aggregates data from multiple sources:

| Source | Description | Records |
|--------|-------------|---------|
| [NBA Box Scores 2010-2024](https://github.com/NocturneBear/NBA-Data-2010-2024) | Historical game-by-game player statistics | ~140,000 games |
| [NBA Player Stats 2024-25](https://www.kaggle.com/datasets/eduardopalmieri/nba-player-stats-season-2425) | Current 2024-25 season box scores (updated regularly) | ~11,000+ games |
| Position Data CSVs | Player position classifications (PG, SG, SF, PF, C) | 550+ players |

### 2. Data Cleaning & Preprocessing

**Notebook:** `NBA_PrizePicks.ipynb`

Key preprocessing steps:

```
Raw Data → Filter Recent Games → Standardize Names → Map Positions → Export Clean CSV
```

- **Temporal Filtering:** Filtered historical data to games from 2023 onwards for relevance
- **Date Parsing:** Converted date strings to datetime objects for time-series analysis
- **Position Mapping:** Cross-referenced 5 separate position datasets to create unified player-position mappings
- **Name Standardization:** Handled "Last, First" → "First Last" format conversions
- **Encoding Issues:** Resolved UTF-8/Latin-1 encoding for international player names (Jokic, Vucevic, etc.)

**Output:** `position_data.csv` - Clean player roster with position classifications

### 3. Feature Engineering

**Notebook:** `Picks_Report.ipynb`

Created derived statistics for PrizePicks prop types:

| Feature | Formula | Use Case |
|---------|---------|----------|
| `PTS+REB` | Points + Rebounds | Combo props |
| `PTS+AST` | Points + Assists | Combo props |
| `PTS+REB+AST` | Points + Rebounds + Assists | PRA props |
| `REB+AST` | Rebounds + Assists | Combo props |
| `STL+BLK` | Steals + Blocks | Defensive props |

### 4. Statistical Analysis

The model computes multiple analytical dimensions:

#### A. Hit Rate Analysis
```python
over_rate = (games where stat > line) / total_games × 100
under_rate = (games where stat < line) / total_games × 100
```

#### B. Recent Form Analysis
- **Rolling Averages:** Last 3, 5, and 10 game performance
- **Trend Detection:** Compares first-half vs second-half of season
- **Streak Tracking:** Consecutive games over/under the line
- **Consistency Score:** Coefficient of variation (lower variance = more predictable)

```python
consistency = 1 - (std_dev / mean)  # Higher = more consistent
```

#### C. Opponent Defense Ranking
- Ranks all 30 teams by average stat allowed
- Position-specific defensive analysis (e.g., how team defends PGs for assists)
- Calculates matchup difficulty percentile

```python
percentile = team_rank / total_teams × 100
# Higher percentile = worse defense = easier matchup
```

#### D. Minutes-Based Comparison
- Groups players by similar minutes played (+/- 3 MPG)
- Ranks player's production against peers with similar workload
- Controls for opportunity when evaluating efficiency

### 5. Confidence Model

The recommendation engine uses a **weighted multi-factor model**:

```
Final Confidence = Σ(Factor Weight × Factor Score) × 100
```

| Factor | Weight | Description |
|--------|--------|-------------|
| Hit Rate | 40% | Season-long over/under success rate |
| Recent Form | 25% | Last 5-10 games, trend, streak, consistency |
| Opponent Defense | 15% | Team defensive ranking + position matchup |
| Sample Size | 10% | Games played (more data = higher confidence) |
| Minutes Comparison | 10% | Performance vs similar-minutes players |

#### Recommendation Thresholds

| Confidence | Recommendation |
|------------|----------------|
| ≥ 60% | **OVER** or **UNDER** (strong play) |
| 45-59% | **LEAN OVER** or **LEAN UNDER** (moderate play) |
| < 45% | **SKIP** (insufficient edge) |

### 6. Visualization

The tool generates two key visualizations per player:

1. **Time Series Plot:** Performance over time with line overlay, showing games above (green) and below (red) the betting line
2. **Distribution Histogram:** Frequency distribution of stat with line and average markers

### 7. Model Validation (Backtesting)

**Module:** `backtester.py` | **Notebook:** `Backtest_Analysis.ipynb`

The model's predictive accuracy was validated using a rigorous backtesting methodology:

#### Methodology
```
Full Season Data → Split at Midpoint → Train on First Half → Predict Second Half → Compare to Actuals
```

- **Training Data:** First half of 2024-25 season (Oct 22 - Dec 15, 2024) - 8,125 games
- **Testing Data:** Second half of season (Dec 15, 2024 - Feb 7, 2025) - 8,387 games
- **Predictions Generated:** 200 balanced OVER/UNDER predictions across all stat categories

#### Backtest Results

| Metric | Result |
|--------|--------|
| **Overall Accuracy** | **72.3%** |
| **High Confidence (>=60%)** | **83.1%** (49/59 correct) |
| **Medium Confidence (45-59%)** | 67.4% (87/129 correct) |
| **OVER Predictions** | 71.8% |
| **UNDER Predictions** | 72.7% |

#### Accuracy by Stat Category

| Stat | Accuracy | Sample Size |
|------|----------|-------------|
| AST | **81.8%** | 44 |
| STL+BLK | **79.5%** | 44 |
| 3P | **79.2%** | 24 |
| TRB | 71.4% | 28 |
| PTS+REB+AST | 57.1% | 28 |
| PTS | 50.0% | 20 |

#### Key Findings

- **Profitable Edge:** 72.3% accuracy far exceeds the 52.4% break-even threshold for standard -110 odds
- **High-Confidence Advantage:** Predictions with >=60% confidence hit at 83.1%
- **Best Categories:** AST, STL+BLK, and 3P predictions show highest reliability
- **Balanced Performance:** Both OVER and UNDER predictions perform similarly well

#### Running the Backtest

```python
from backtester import PredictionBacktester

# Initialize and run backtest
backtester = PredictionBacktester()
results = backtester.run_backtest(n_predictions=200)

# Results saved to backtest_results.csv
```

---

## Project Structure

```
Young-Bull-Picks/
├── NBA_PrizePicks.ipynb      # Data cleaning & preprocessing
├── Picks_Report.ipynb        # Analysis & recommendation engine
├── Backtest_Analysis.ipynb   # Interactive backtest visualization
├── backtester.py             # Backtesting module for model validation
├── position_data.csv         # Processed player-position mappings
├── backtest_results.csv      # Detailed backtest prediction results
└── README.md                 # Project documentation
```

---

## Usage

### Quick Start

1. Open `Picks_Report.ipynb` in Jupyter
2. Run all cells to load data and functions
3. Enter your picks:

```python
YOUR_PICKS = [
    ("LeBron James", "PTS", 25.5, "GSW"),
    ("Nikola Jokic", "AST", 8.5, "LAL"),
    ("Jayson Tatum", "PTS+REB+AST", 45.5, None),
]

results = analyze_multiple_picks(YOUR_PICKS)
```

### Detailed Analysis

```python
# Full report with all factors
full_report("Stephen Curry", "3P", 4.5, opponent="LAL")
```

### Find Best Plays

```python
# Find players with 70%+ hit rates
best_picks = find_best_picks('PTS', min_games=20, min_hit_rate=70)
```

---

## Key Functions

### Analysis Functions (`Picks_Report.ipynb`)

| Function | Purpose |
|----------|---------|
| `analyze_line()` | Calculate hit rates and basic stats for a line |
| `analyze_recent_form()` | Detailed recent performance breakdown |
| `get_opponent_defensive_ranking()` | Team defense ranking for a stat |
| `get_opponent_defense_vs_position()` | Position-specific defensive analysis |
| `get_similar_minutes_comparison()` | Compare to players with similar workload |
| `generate_recommendation()` | Multi-factor confidence calculation |
| `full_report()` | Complete analysis with visualizations |
| `analyze_multiple_picks()` | Batch analysis for multiple props |
| `find_best_picks()` | Discover high-value plays automatically |

### Backtesting Functions (`backtester.py`)

| Function | Purpose |
|----------|---------|
| `PredictionBacktester()` | Initialize backtester with data loading and splitting |
| `run_backtest()` | Execute full backtest pipeline with N predictions |
| `generate_test_predictions()` | Generate predictions using training data only |
| `evaluate_predictions()` | Calculate accuracy metrics and breakdowns |
| `print_results()` | Display formatted backtest results summary |

---

## Tech Stack

- **Python 3.11+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib/Seaborn** - Visualization
- **Jupyter Notebook** - Interactive development
- **KaggleHub** - Dataset management

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ElanHashem/Young-Bull-Picks.git
cd Young-Bull-Picks

# Install dependencies
pip install pandas numpy matplotlib seaborn kagglehub jupyter

# Launch Jupyter
jupyter notebook Picks_Report.ipynb
```

---

## Disclaimer

This tool is for educational and entertainment purposes only. Sports betting involves risk, and past performance does not guarantee future results. Always gamble responsibly.

---

## Author

**Elan Hashem**

*Built with data science and a love for basketball analytics.*

"""
Backtesting Module for Young Bull Picks Prediction Model

This module tests the accuracy of the prediction model by:
1. Using the first half of the 24/25 season as training data
2. Making predictions for games in the second half
3. Comparing predictions against actual results
"""

import pandas as pd
import numpy as np
import os
import ast
import kagglehub
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PredictionBacktester:
    """
    Backtesting class that validates the prediction model accuracy
    by testing on held-out data from the second half of the season.
    """

    def __init__(self):
        """Initialize the backtester with data loading."""
        self.load_data()
        self.split_data()

    def load_data(self):
        """Load season data and position data."""
        # Load position data
        csv_path = os.path.join(os.path.dirname(__file__), "position_data.csv")
        self.players = pd.read_csv(csv_path, encoding='latin-1')

        # Parse position column
        def parse_position(pos_str):
            if pd.isna(pos_str) or pos_str == '[]':
                return []
            try:
                return ast.literal_eval(pos_str)
            except:
                return []

        self.players['Position'] = self.players['Position'].apply(parse_position)

        # Load season data from Kaggle
        path = kagglehub.dataset_download("eduardopalmieri/nba-player-stats-season-2425")
        csv_file = os.path.join(path, "database_24_25.csv")
        self.full_data = pd.read_csv(csv_file)

        # Convert date column
        self.full_data['Data'] = pd.to_datetime(self.full_data['Data'])

        # Create combined stats
        self.full_data['PTS+REB'] = self.full_data['PTS'] + self.full_data['TRB']
        self.full_data['PTS+AST'] = self.full_data['PTS'] + self.full_data['AST']
        self.full_data['PTS+REB+AST'] = self.full_data['PTS'] + self.full_data['TRB'] + self.full_data['AST']
        self.full_data['REB+AST'] = self.full_data['TRB'] + self.full_data['AST']
        self.full_data['STL+BLK'] = self.full_data['STL'] + self.full_data['BLK']

        print(f"Loaded {len(self.full_data)} total game records")
        print(f"Date range: {self.full_data['Data'].min()} to {self.full_data['Data'].max()}")

    def split_data(self):
        """Split data into training (first half) and testing (second half) sets."""
        # Find the midpoint date
        min_date = self.full_data['Data'].min()
        max_date = self.full_data['Data'].max()
        mid_date = min_date + (max_date - min_date) / 2

        self.training_data = self.full_data[self.full_data['Data'] < mid_date].copy()
        self.testing_data = self.full_data[self.full_data['Data'] >= mid_date].copy()

        print(f"\nData Split:")
        print(f"Training data: {len(self.training_data)} games ({min_date.date()} to {mid_date.date()})")
        print(f"Testing data: {len(self.testing_data)} games ({mid_date.date()} to {max_date.date()})")

        # Store the split date for reference
        self.split_date = mid_date

    def get_player_positions(self, player_name):
        """Get positions for a player from position data."""
        match = self.players[self.players['Name'] == player_name]
        if len(match) > 0:
            return match.iloc[0]['Position']
        return []

    def analyze_line_training(self, player_name, stat_col, line):
        """
        Analyze a betting line using only training data.
        Returns hit rate, average, and other metrics.
        """
        player_data = self.training_data[self.training_data['Player'] == player_name].copy()
        player_data = player_data.sort_values('Data', ascending=False)

        if len(player_data) == 0:
            return None

        stats = player_data[stat_col]

        over_hits = (stats > line).sum()
        under_hits = (stats < line).sum()
        pushes = (stats == line).sum()
        total_games = len(stats)

        return {
            'player': player_name,
            'stat': stat_col,
            'line': line,
            'games_analyzed': total_games,
            'average': stats.mean(),
            'median': stats.median(),
            'std_dev': stats.std(),
            'min': stats.min(),
            'max': stats.max(),
            'over_hits': over_hits,
            'under_hits': under_hits,
            'pushes': pushes,
            'over_rate': over_hits / total_games * 100 if total_games > 0 else 0,
            'under_rate': under_hits / total_games * 100 if total_games > 0 else 0,
            'last_5_avg': player_data.head(5)[stat_col].mean() if len(player_data) >= 5 else stats.mean(),
            'last_10_avg': player_data.head(10)[stat_col].mean() if len(player_data) >= 10 else stats.mean(),
            'avg_minutes': player_data['MP'].mean(),
        }

    def analyze_recent_form_training(self, player_name, stat_col, line):
        """Detailed analysis of recent performance trends using training data."""
        player_data = self.training_data[self.training_data['Player'] == player_name].copy()
        player_data = player_data.sort_values('Data', ascending=False)

        if len(player_data) < 5:
            return None

        season_avg = player_data[stat_col].mean()

        # Last 3, 5, 10 games
        last_3 = player_data.head(3)[stat_col]
        last_5 = player_data.head(5)[stat_col]
        last_10 = player_data.head(10)[stat_col] if len(player_data) >= 10 else player_data[stat_col]

        # Hit rates for different windows
        last_5_over_rate = (last_5 > line).mean() * 100
        last_10_over_rate = (last_10 > line).mean() * 100

        # Trend detection
        if len(player_data) >= 10:
            first_half = player_data.tail(len(player_data)//2)[stat_col].mean()
            second_half = player_data.head(len(player_data)//2)[stat_col].mean()
            trend = second_half - first_half
            trend_pct = (trend / first_half * 100) if first_half > 0 else 0
        else:
            trend = 0
            trend_pct = 0

        # Streak detection
        streak = 0
        streak_type = None
        for val in player_data[stat_col].values:
            if streak == 0:
                if val > line:
                    streak = 1
                    streak_type = 'OVER'
                else:
                    streak = 1
                    streak_type = 'UNDER'
            elif streak_type == 'OVER' and val > line:
                streak += 1
            elif streak_type == 'UNDER' and val < line:
                streak += 1
            else:
                break

        # Consistency score
        consistency = 1 - (player_data[stat_col].std() / player_data[stat_col].mean()) if player_data[stat_col].mean() > 0 else 0
        consistency = max(0, min(1, consistency))

        return {
            'player': player_name,
            'stat': stat_col,
            'line': line,
            'season_avg': season_avg,
            'last_3_avg': last_3.mean(),
            'last_5_avg': last_5.mean(),
            'last_10_avg': last_10.mean(),
            'last_5_over_rate': last_5_over_rate,
            'last_10_over_rate': last_10_over_rate,
            'trend': trend,
            'trend_pct': trend_pct,
            'trend_direction': 'UP' if trend > 0 else 'DOWN' if trend < 0 else 'FLAT',
            'current_streak': streak,
            'streak_type': streak_type,
            'consistency_score': consistency * 100,
        }

    def get_opponent_defensive_ranking_training(self, opponent, stat_col):
        """Calculate opponent defensive ranking using training data only."""
        team_defense = self.training_data.groupby('Opp')[stat_col].agg(['mean', 'count']).reset_index()
        team_defense.columns = ['Team', 'Avg_Allowed', 'Games']

        team_defense = team_defense[team_defense['Games'] >= 5]

        if len(team_defense) == 0:
            return None

        team_defense['Rank'] = team_defense['Avg_Allowed'].rank(ascending=True)
        team_defense['Percentile'] = team_defense['Avg_Allowed'].rank(pct=True) * 100

        opp_row = team_defense[team_defense['Team'] == opponent]

        if len(opp_row) == 0:
            return None

        league_avg = team_defense['Avg_Allowed'].mean()

        return {
            'opponent': opponent,
            'stat': stat_col,
            'avg_allowed': opp_row['Avg_Allowed'].values[0],
            'rank': int(opp_row['Rank'].values[0]),
            'total_teams': len(team_defense),
            'percentile': opp_row['Percentile'].values[0],
            'league_avg': league_avg,
        }

    def get_opponent_defense_vs_position_training(self, opponent, stat_col, positions):
        """Calculate opponent defense vs position using training data."""
        if not positions:
            return None

        position_players = self.players[
            self.players['Position'].apply(lambda x: any(pos in x for pos in positions))
        ]['Name'].tolist()

        vs_opp = self.training_data[
            (self.training_data['Player'].isin(position_players)) &
            (self.training_data['Opp'] == opponent)
        ]

        vs_all = self.training_data[self.training_data['Player'].isin(position_players)]

        if len(vs_opp) < 3:
            return None

        opp_avg = vs_opp[stat_col].mean()
        league_avg = vs_all[stat_col].mean()

        return {
            'opponent': opponent,
            'positions': positions,
            'stat': stat_col,
            'avg_allowed': opp_avg,
            'league_avg': league_avg,
            'differential': opp_avg - league_avg,
        }

    def get_similar_minutes_comparison_training(self, player_name, stat_col, minutes_range=3):
        """Compare player's stats to similar-minutes players using training data."""
        player_data = self.training_data[self.training_data['Player'] == player_name]

        if len(player_data) == 0:
            return None

        player_avg_minutes = player_data['MP'].mean()
        player_avg_stat = player_data[stat_col].mean()

        player_avgs = self.training_data.groupby('Player').agg({
            'MP': 'mean',
            stat_col: 'mean',
            'Player': 'count'
        }).rename(columns={'Player': 'Games'}).reset_index()

        similar = player_avgs[
            (player_avgs['MP'] >= player_avg_minutes - minutes_range) &
            (player_avgs['MP'] <= player_avg_minutes + minutes_range) &
            (player_avgs['Games'] >= 5)
        ].copy()

        if len(similar) < 5:
            return None

        similar['Rank'] = similar[stat_col].rank(ascending=False)
        similar['Percentile'] = similar[stat_col].rank(pct=True) * 100

        player_row = similar[similar['Player'] == player_name]

        if len(player_row) == 0:
            return None

        return {
            'player': player_name,
            'stat': stat_col,
            'player_avg': player_avg_stat,
            'player_minutes': player_avg_minutes,
            'similar_players_avg': similar[stat_col].mean(),
            'rank_among_similar': int(player_row['Rank'].values[0]),
            'total_similar_players': len(similar),
            'percentile': player_row['Percentile'].values[0],
        }

    def generate_prediction(self, player_name, stat_col, line, opponent=None):
        """
        Generate a betting prediction based on training data only.
        Uses the same multi-factor weighted model as the main notebook.

        Returns: (direction, confidence, factors)
        """
        analysis = self.analyze_line_training(player_name, stat_col, line)

        if analysis is None:
            return None, 0, {}

        over_rate = analysis['over_rate']
        under_rate = analysis['under_rate']

        factors = {}

        # FACTOR 1: Base Hit Rate (40% weight)
        if over_rate >= 70:
            base_direction = "OVER"
            base_confidence = (over_rate - 50) / 50
        elif under_rate >= 70:
            base_direction = "UNDER"
            base_confidence = (under_rate - 50) / 50
        elif over_rate >= 55:
            base_direction = "LEAN OVER"
            base_confidence = (over_rate - 50) / 50 * 0.7
        elif under_rate >= 55:
            base_direction = "LEAN UNDER"
            base_confidence = (under_rate - 50) / 50 * 0.7
        else:
            base_direction = "SKIP"
            base_confidence = 0

        factors['hit_rate'] = {
            'weight': 0.40,
            'score': base_confidence,
            'detail': f"{over_rate:.1f}% over / {under_rate:.1f}% under"
        }

        # FACTOR 2: Sample Size (10% weight)
        sample_score = min(analysis['games_analyzed'] / 20, 1.0)
        factors['sample_size'] = {
            'weight': 0.10,
            'score': sample_score,
            'detail': f"{analysis['games_analyzed']} games"
        }

        # FACTOR 3: Recent Form (25% weight)
        recent_form = self.analyze_recent_form_training(player_name, stat_col, line)
        if recent_form:
            if base_direction in ["OVER", "LEAN OVER"]:
                recent_over_rate = recent_form['last_5_over_rate']
                trend_helps = recent_form['trend_direction'] == 'UP'
                streak_helps = recent_form['streak_type'] == 'OVER'
            else:
                recent_over_rate = 100 - recent_form['last_5_over_rate']
                trend_helps = recent_form['trend_direction'] == 'DOWN'
                streak_helps = recent_form['streak_type'] == 'UNDER'

            recent_score = recent_over_rate / 100
            trend_modifier = 0.1 if trend_helps else -0.1 if recent_form['trend_direction'] != 'FLAT' else 0
            streak_modifier = min(recent_form['current_streak'] * 0.02, 0.1) if streak_helps else 0
            consistency_modifier = (recent_form['consistency_score'] / 100) * 0.1

            form_score = min(1, max(0, recent_score + trend_modifier + streak_modifier + consistency_modifier))

            factors['recent_form'] = {
                'weight': 0.25,
                'score': form_score,
                'detail': f"L5: {recent_form['last_5_avg']:.1f}, Trend: {recent_form['trend_direction']}"
            }
        else:
            factors['recent_form'] = {'weight': 0.25, 'score': 0.5, 'detail': 'Insufficient data'}

        # FACTOR 4: Opponent Defense (15% weight)
        if opponent:
            def_ranking = self.get_opponent_defensive_ranking_training(opponent, stat_col)
            positions = self.get_player_positions(player_name)
            pos_defense = self.get_opponent_defense_vs_position_training(opponent, stat_col, positions) if positions else None

            if def_ranking:
                if base_direction in ["OVER", "LEAN OVER"]:
                    defense_score = def_ranking['percentile'] / 100
                else:
                    defense_score = 1 - (def_ranking['percentile'] / 100)

                if pos_defense:
                    pos_diff = pos_defense['differential']
                    if base_direction in ["OVER", "LEAN OVER"] and pos_diff > 0:
                        defense_score = min(1, defense_score + 0.1)
                    elif base_direction in ["UNDER", "LEAN UNDER"] and pos_diff < 0:
                        defense_score = min(1, defense_score + 0.1)

                factors['opponent_defense'] = {
                    'weight': 0.15,
                    'score': defense_score,
                    'detail': f"Rank #{def_ranking['rank']}/{def_ranking['total_teams']}"
                }
            else:
                factors['opponent_defense'] = {'weight': 0.15, 'score': 0.5, 'detail': 'No data'}
        else:
            factors['opponent_defense'] = {'weight': 0.15, 'score': 0.5, 'detail': 'No opponent'}

        # FACTOR 5: Minutes Comparison (10% weight)
        minutes_comp = self.get_similar_minutes_comparison_training(player_name, stat_col)
        if minutes_comp:
            if base_direction in ["OVER", "LEAN OVER"]:
                minutes_score = minutes_comp['percentile'] / 100
            else:
                minutes_score = 1 - (minutes_comp['percentile'] / 100)

            factors['minutes_comparison'] = {
                'weight': 0.10,
                'score': minutes_score,
                'detail': f"#{minutes_comp['rank_among_similar']}/{minutes_comp['total_similar_players']}"
            }
        else:
            factors['minutes_comparison'] = {'weight': 0.10, 'score': 0.5, 'detail': 'Insufficient data'}

        # Calculate final confidence
        if base_direction == "SKIP":
            final_confidence = 0
            direction = "SKIP"
        else:
            weighted_sum = sum(f['weight'] * f['score'] for f in factors.values())
            total_weight = sum(f['weight'] for f in factors.values())
            final_confidence = (weighted_sum / total_weight) * 100

            if final_confidence >= 60:
                direction = "OVER" if base_direction in ["OVER", "LEAN OVER"] else "UNDER"
            elif final_confidence >= 45:
                direction = "LEAN OVER" if base_direction in ["OVER", "LEAN OVER"] else "LEAN UNDER"
            else:
                direction = "SKIP"

        return direction, final_confidence, factors

    def generate_test_predictions(self, n_predictions=100, balanced=True):
        """
        Generate N predictions for players in the test set.
        Returns a list of prediction dictionaries with actual outcomes.

        Args:
            n_predictions: Target number of predictions
            balanced: If True, try to balance OVER/UNDER predictions
        """
        predictions = []
        over_count = 0
        under_count = 0

        # Get players who have data in both training and testing sets
        training_players = set(self.training_data['Player'].unique())
        testing_players = set(self.testing_data['Player'].unique())
        common_players = list(training_players & testing_players)

        # Filter to players with significant playing time
        good_players = []
        for player in common_players:
            train_data = self.training_data[self.training_data['Player'] == player]
            if len(train_data) >= 10 and train_data['MP'].mean() >= 15:
                good_players.append(player)

        print(f"\nPlayers with data in both sets: {len(common_players)}")
        print(f"Players with 10+ games and 15+ MPG: {len(good_players)}")

        # Stat categories to test (weighted by importance in betting)
        stat_cols = ['PTS', 'PTS', 'TRB', 'AST', 'AST', '3P', 'PTS+REB+AST', 'PTS+REB+AST', 'STL+BLK']

        # Generate predictions
        np.random.seed(42)  # For reproducibility
        attempts = 0
        max_attempts = n_predictions * 20
        seen_combos = set()

        while len(predictions) < n_predictions and attempts < max_attempts:
            attempts += 1

            # Random player and stat
            player = np.random.choice(good_players)
            stat_col = np.random.choice(stat_cols)

            # Get player's training data
            train_player = self.training_data[self.training_data['Player'] == player]
            if len(train_player) < 10:
                continue

            # Get player's test data
            test_player = self.testing_data[self.testing_data['Player'] == player]
            if len(test_player) < 3:
                continue

            # Calculate line based on training average (simulate realistic line)
            train_avg = train_player[stat_col].mean()
            train_std = train_player[stat_col].std()

            # Vary the line around the average to get both OVER and UNDER opportunities
            line_offset = np.random.choice([-1.0, -0.5, 0, 0.5, 1.0])
            line = round((train_avg + line_offset) * 2) / 2  # Round to nearest 0.5

            # Get a random test game for this player
            test_game = test_player.sample(1).iloc[0]
            opponent = test_game['Opp']
            actual_value = test_game[stat_col]
            game_date = test_game['Data']

            # Skip if we've seen this exact combo
            combo_key = (player, stat_col, line, str(game_date))
            if combo_key in seen_combos:
                continue
            seen_combos.add(combo_key)

            # Generate prediction
            direction, confidence, factors = self.generate_prediction(player, stat_col, line, opponent)

            if direction == "SKIP":
                continue

            # Balance OVER/UNDER if requested
            if balanced:
                is_over = direction in ["OVER", "LEAN OVER"]
                target_each = n_predictions // 2
                if is_over and over_count >= target_each + 10:
                    continue
                if not is_over and under_count >= target_each + 10:
                    continue

            # Determine actual outcome
            if actual_value > line:
                actual_outcome = "OVER"
            elif actual_value < line:
                actual_outcome = "UNDER"
            else:
                actual_outcome = "PUSH"

            # Determine if prediction was correct
            if actual_outcome == "PUSH":
                correct = None
            elif direction in ["OVER", "LEAN OVER"]:
                correct = actual_outcome == "OVER"
                over_count += 1
            else:
                correct = actual_outcome == "UNDER"
                under_count += 1

            predictions.append({
                'player': player,
                'stat': stat_col,
                'line': line,
                'opponent': opponent,
                'game_date': game_date,
                'prediction': direction,
                'confidence': confidence,
                'actual_value': actual_value,
                'actual_outcome': actual_outcome,
                'correct': correct,
                'margin': actual_value - line
            })

        print(f"Generated {len(predictions)} predictions after {attempts} attempts")
        print(f"OVER predictions: {over_count}, UNDER predictions: {under_count}")
        return predictions

    def evaluate_predictions(self, predictions):
        """
        Evaluate the accuracy of predictions.
        Returns detailed metrics and breakdown.
        """
        df = pd.DataFrame(predictions)

        # Remove pushes for accuracy calculation
        df_no_push = df[df['actual_outcome'] != 'PUSH']

        # Overall accuracy
        overall_accuracy = df_no_push['correct'].mean() * 100

        # Accuracy by confidence level
        high_conf = df_no_push[df_no_push['confidence'] >= 60]
        med_conf = df_no_push[(df_no_push['confidence'] >= 45) & (df_no_push['confidence'] < 60)]

        high_conf_accuracy = high_conf['correct'].mean() * 100 if len(high_conf) > 0 else 0
        med_conf_accuracy = med_conf['correct'].mean() * 100 if len(med_conf) > 0 else 0

        # Accuracy by direction
        overs = df_no_push[df_no_push['prediction'].isin(['OVER', 'LEAN OVER'])]
        unders = df_no_push[df_no_push['prediction'].isin(['UNDER', 'LEAN UNDER'])]

        over_accuracy = overs['correct'].mean() * 100 if len(overs) > 0 else 0
        under_accuracy = unders['correct'].mean() * 100 if len(unders) > 0 else 0

        # Accuracy by stat category
        stat_accuracy = df_no_push.groupby('stat')['correct'].agg(['mean', 'count'])
        stat_accuracy['accuracy'] = stat_accuracy['mean'] * 100

        results = {
            'total_predictions': len(df),
            'pushes': len(df[df['actual_outcome'] == 'PUSH']),
            'evaluated_predictions': len(df_no_push),
            'overall_accuracy': overall_accuracy,
            'correct_predictions': int(df_no_push['correct'].sum()),
            'incorrect_predictions': len(df_no_push) - int(df_no_push['correct'].sum()),
            'high_confidence': {
                'count': len(high_conf),
                'accuracy': high_conf_accuracy,
                'correct': int(high_conf['correct'].sum()) if len(high_conf) > 0 else 0
            },
            'medium_confidence': {
                'count': len(med_conf),
                'accuracy': med_conf_accuracy,
                'correct': int(med_conf['correct'].sum()) if len(med_conf) > 0 else 0
            },
            'over_predictions': {
                'count': len(overs),
                'accuracy': over_accuracy,
                'correct': int(overs['correct'].sum()) if len(overs) > 0 else 0
            },
            'under_predictions': {
                'count': len(unders),
                'accuracy': under_accuracy,
                'correct': int(unders['correct'].sum()) if len(unders) > 0 else 0
            },
            'by_stat': stat_accuracy.to_dict(),
            'predictions_df': df
        }

        return results

    def print_results(self, results):
        """Print formatted results summary."""
        print("\n" + "="*70)
        print("BACKTEST RESULTS - PREDICTION MODEL ACCURACY")
        print("="*70)

        print(f"\n--- OVERALL PERFORMANCE ---")
        print(f"Total Predictions: {results['total_predictions']}")
        print(f"Pushes (excluded): {results['pushes']}")
        print(f"Evaluated: {results['evaluated_predictions']}")
        print(f"\nOVERALL ACCURACY: {results['overall_accuracy']:.1f}%")
        print(f"Correct: {results['correct_predictions']}")
        print(f"Incorrect: {results['incorrect_predictions']}")

        print(f"\n--- BY CONFIDENCE LEVEL ---")
        hc = results['high_confidence']
        mc = results['medium_confidence']
        print(f"High Confidence (>=60%): {hc['accuracy']:.1f}% ({hc['correct']}/{hc['count']})")
        print(f"Medium Confidence (45-59%): {mc['accuracy']:.1f}% ({mc['correct']}/{mc['count']})")

        print(f"\n--- BY DIRECTION ---")
        ov = results['over_predictions']
        un = results['under_predictions']
        print(f"OVER predictions: {ov['accuracy']:.1f}% ({ov['correct']}/{ov['count']})")
        print(f"UNDER predictions: {un['accuracy']:.1f}% ({un['correct']}/{un['count']})")

        print(f"\n--- BY STAT CATEGORY ---")
        stat_data = results['by_stat']
        for stat in stat_data['accuracy'].keys():
            acc = stat_data['accuracy'][stat]
            count = stat_data['count'][stat]
            print(f"{stat:15}: {acc:.1f}% ({int(count)} predictions)")

        print("\n" + "="*70)

        # Show some example predictions
        df = results['predictions_df']
        print("\n--- SAMPLE PREDICTIONS ---")
        sample = df.sample(min(10, len(df)))[['player', 'stat', 'line', 'prediction',
                                              'confidence', 'actual_value', 'actual_outcome', 'correct']]
        print(sample.to_string(index=False))

        return results

    def run_backtest(self, n_predictions=100, balanced=True):
        """
        Run the full backtest pipeline.

        Args:
            n_predictions: Number of predictions to generate and test
            balanced: If True, try to balance OVER/UNDER predictions

        Returns:
            Dictionary with all results and metrics
        """
        print("\n" + "="*70)
        print("STARTING BACKTEST")
        print("="*70)
        print(f"\nGenerating {n_predictions} predictions using training data...")
        print("Testing against held-out second-half season data...")

        predictions = self.generate_test_predictions(n_predictions, balanced=balanced)
        results = self.evaluate_predictions(predictions)
        self.print_results(results)

        return results


def main():
    """Run the backtest."""
    backtester = PredictionBacktester()

    # Run with 200 predictions for better statistical significance
    results = backtester.run_backtest(n_predictions=200)

    # Save results to CSV
    df = results['predictions_df']
    output_path = os.path.join(os.path.dirname(__file__), "backtest_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

    # Print key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    accuracy = results['overall_accuracy']
    if accuracy >= 55:
        print(f"[+] Model shows positive edge at {accuracy:.1f}% accuracy")
        print("    - Profitable threshold is typically 52.4% (accounting for -110 odds)")
    else:
        print(f"[-] Model accuracy ({accuracy:.1f}%) is below profitable threshold")

    hc = results['high_confidence']
    if hc['accuracy'] > accuracy and hc['count'] >= 10:
        print(f"[+] High confidence picks ({hc['accuracy']:.1f}%) outperform overall")
        print(f"    - Focus on picks with >=60% confidence for best results")

    # Best stat categories
    stat_data = results['by_stat']
    best_stat = max(stat_data['accuracy'].keys(), key=lambda x: stat_data['accuracy'][x])
    best_acc = stat_data['accuracy'][best_stat]
    print(f"[+] Best performing stat: {best_stat} ({best_acc:.1f}%)")

    return results


if __name__ == "__main__":
    main()

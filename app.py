import os
import re
import pandas as pd
import numpy as np
import datetime
import openai
import logging
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Initialize Flask app
app = Flask(__name__)

# Load OpenAI API key (Replace with your API key)
openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    logger.error("OPENAI_API_KEY environment variable not set.")


class PRAnalytics:
    """Class to handle PR dataset analytics and predictions."""

    def __init__(self, file_path):
        self.df = self.load_data(file_path)
        self.model, self.scaler = self.train_merge_time_model()

    def load_data(self, file_path):
        try:
            """Load and preprocess PR dataset."""
            df = pd.read_excel(file_path)
            df['Created_Date'] = pd.to_datetime(df['Created_Date'])
            df['Merged_Date'] = pd.to_datetime(df['Merged_Date'])

            # Compute merge time (days)
            df['Merge_Time'] = (df['Merged_Date'] - df['Created_Date']).dt.days
            df['Merge_Time'] = df['Merge_Time'].fillna(-1)  # -1 for open PRs
            return df
        except FileNotFoundError:
            logger.error(f"Error: Data file '{file_path}' not found.")
            return pd.DataFrame()  # Return empty DataFrame on error
        except Exception as e:
            logger.exception(f"An unexpected error occurred while loading data: {e}")
            return pd.DataFrame()

    def average_merge_time(self, days=30):
        try:
            """Calculate the average merge time for PRs in the last `days` days."""
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_df = self.df[(self.df['Merged_Date'].notna()) & (self.df['Created_Date'] >= cutoff_date)]

            if filtered_df.empty:
                return "No PRs were merged in the selected timeframe."

            return f"The average PR merge time in the last {days} days is {filtered_df['Merge_Time'].mean():.2f} days."
        except Exception as e:
            logger.exception(f"Error calculating average merge time: {e}")
            return "An error occurred while calculating average merge time."

    def find_bottlenecks(self):
        try:
            """Identify PRs that took longer than the average to merge."""
            df = self.df[self.df['Merged_Date'].notna()]
            avg_time = df['Merge_Time'].mean()
            bottleneck_prs = df[df['Merge_Time'] > avg_time]

            if bottleneck_prs.empty:
                return "No bottleneck PRs identified."

            return bottleneck_prs[['PR_ID', 'Assignee', 'Status', 'Merge_Time']].to_dict(orient="records")
        except Exception as e:
            logger.exception(f"Error calculating bottle necks: {e}")
            return "An error occurred while calculating for finding bottlenecks."

    def find_delayed_prs(self, days=30, threshold_multiplier=1.5):
        try:# Added threshold customization
            if self.df.empty:
                return "Error: No data loaded."
            today = datetime.today()
            cutoff = today - timedelta(days=days)
            subset = self.df[(self.df['Created_Date'] >= cutoff) & (self.df['Status'] == 'merged')]
            average = self.average_merge_time(days)
            if isinstance(average, str):
                return average  # Handle error from average_merge_time
            threshold = average * threshold_multiplier  # Customizable threshold

            delayed = subset[subset['Merge_Time'] > threshold]
            return delayed[['PR_ID', 'Created_Date', 'Merged_Date', 'Assignee', 'Status', 'Reviewer_Feedback']]
        except Exception as e:
            logger.exception(f"Error calculating find delayed PRs: {e}")
            return "An error occurred while calculating find delayed PRs."

    def most_delayed_developer(self, days=30):
        try:# Added threshold customization
            if self.df.empty:
                return "Error: No data loaded."
            today = datetime.today()
            cutoff = today - timedelta(days=days)
            merged_prs = self.df[
                (self.df['Merged_Date'].notna()) & (self.df['Created_Date'] >= cutoff) & (self.df['Status'] == 'merged')]

            if merged_prs.empty:
                return "No merged PRs found within the specified timeframe."

            average_merge_time = merged_prs['Merge_Time'].mean()
            delayed_prs = merged_prs[merged_prs['Merge_Time'] > average_merge_time]

            if delayed_prs.empty:
                return "No PRs were found to be delayed within the specified timeframe."

            developer_counts = delayed_prs['Assignee'].value_counts()
            most_delayed_dev = developer_counts.idxmax()
            return most_delayed_dev, delayed_prs
        except Exception as e:
            logger.exception(f"Error calculating most delayed developer: {e}")
            return "An error occurred while calculating most delayed developer."

    def pr_throughput_trend(self, weeks=3):
        try:
            """Compute PR throughput trends over the last `weeks` weeks."""
            df = self.df[self.df['Merged_Date'].notna()]
            df['Week'] = df['Merged_Date'].dt.to_period('W')
            trend = df.groupby('Week').size().reset_index(name='PR_Count')

            if trend.empty:
                return "No PR throughput data available."
            weeks_data = trend['PR_Count'].tolist()
            response = f"PR throughput over the last {weeks} weeks:\n"
            for i, count in enumerate(weeks_data):
                week_ago = datetime.now() - timedelta(weeks=weeks - 1 - i)  # Corrected week calculation
                start_week = week_ago - timedelta(days=week_ago.weekday())  # Start of the week
                end_week = start_week + timedelta(days=6)  # End of the week
                response += f"  Week of {start_week.strftime('%Y-%m-%d')} - {end_week.strftime('%Y-%m-%d')}: {count} PRs\n"
            return response
        except Exception as e:
            logger.exception(f"Error calculating PR throughput trend: {e}")
            return "An error occurred while calculating PR Throughput trend."

    def train_merge_time_model(self):
        try:
            """Train a simple linear regression model to predict merge time."""
            df = self.df[self.df['Merge_Time'] >= 0]  # Only use merged PRs
            # Simple feature: PR index
            X = np.arange(len(df)).reshape(-1, 1)
            y = df['Merge_Time'].values
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # Standardize the features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            # Evaluate model
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            return model, scaler
        except Exception as e:
            logger.exception(f"Error calculating train merge time model : {e}")
            return "An error occurred while calculating train merge time model."

    def predict_merge_time(self, num_prs):
        try:
            """Predict merge time based on PR count using the trained model."""
            X_new = np.array([[num_prs]])  # Feature
            X_new_scaled = self.scaler.transform(X_new)
            predicted_days = self.model.predict(X_new_scaled)[0]
            return max(0, round(predicted_days))
        except Exception as e:
            logger.exception(f"Error calculating predict merge time: {e}")
            return "An error occurred while calculating predict merge time."

    def predict_bottlenecks(self, num_prs):
        try:
            """Predict if bottlenecks are likely based on predicted merge time for a given number of PRs."""
            predicted_time = self.predict_merge_time(num_prs)
            # Use current average merge time from merged PRs
            current_avg = self.df[self.df['Merged_Date'].notna()]['Merge_Time'].mean()
            if predicted_time >= current_avg:
                return f"With a predicted merge time of {num_prs} PRs (above the current average of {current_avg:.2f} days), bottlenecks are likely."
            else:
                return f"With a predicted merge time of {predicted_time} PRs (below the current average of {current_avg:.2f} days), bottlenecks are unlikely."
        except Exception as e:
            logger.exception(f"Error calculating predict_bottleneck: {e}")
            return "An error occurred while calculating Predict_bottlenecks method."

    def predict_throughput_trend(self, future_weeks=1):
        try:
            df = self.df[self.df['Merged_Date'].notna()].copy()
            if df.empty:
                return "No PR throughput data available for prediction."

            df['Week'] = df['Merged_Date'].dt.to_period('W').astype(str)
            trend = df.groupby('Week').size().reset_index(name='PR_Count')
            # Check for sufficient data before model training
            if trend.shape[0] < 2:
                return "Insufficient data to train a reliable model.  Please provide more data."

            trend['WeekStart'] = pd.to_datetime(trend['Week'] + '-1', format='%Y-W%W-%w', errors='coerce')
            trend.dropna(subset=['WeekStart'], inplace=True)
            trend['WeekOrdinal'] = trend['WeekStart'].apply(lambda x: x.toordinal())

            X = trend[['WeekOrdinal']]
            y = trend['PR_Count']

            reg = LinearRegression()
            reg.fit(X, y)

            last_week_date = trend['WeekStart'].max()
            predictions = {}
            for i in range(1, future_weeks + 1):
                future_date = last_week_date + timedelta(weeks=i)
                print(future_date)
                future_ordinal = future_date.toordinal()
                print(future_ordinal)
                predicted_count = reg.predict([[future_ordinal]])[0]
                print(predicted_count)
                predictions[future_date.strftime("%Y-%m-%d")] = round(predicted_count)
            return predictions

        except Exception as e:
            logger.exception(f"An error occurred in predict_throughput_trend: {e}")
            return f"An error occurred while predicting the trend: {e}"


# Initialize PR Analytics instance (ensure your Excel file is in the same directory)
pr_analytics = PRAnalytics("Pull_Request_Data.xlsx")


class ChatGPTHandler:
    """Class to handle responses: either using predictions from the trained model or OpenAI's API."""

    @staticmethod
    def query_chatgpt(prompt):
        try:
            lower_prompt = prompt.lower()

            if "predict average merge time" in lower_prompt:
                match = re.search(r"(\d+)", prompt)
                if match:
                    num_prs = int(match.group(1))
                    prediction = pr_analytics.predict_merge_time(num_prs)
                    return f"Predicted average merge time for {num_prs} PR(s) is approximately {prediction} days."
                else:
                    return "Please specify the number of PRs for which you want to predict average merge time."
            elif re.search(r"average PR merge time for last month", prompt, re.IGNORECASE):
                average = pr_analytics.average_merge_time(days=30)  # Last month is approximately 30 days
                return average

            elif "predict bottlenecks" in lower_prompt:
                match = re.search(r"(\d+)", prompt)
                if match:
                    num_prs = int(match.group(1))
                    prediction = pr_analytics.predict_bottlenecks(num_prs)
                    return prediction
                else:
                    return "Please specify the number of PRs for bottleneck prediction."


            elif re.search(r"which developer has the most delayed prs", prompt, re.IGNORECASE):

                days = 30  # default to 30 days if not specified

                match = re.search(r"(\d+)", prompt)  # Check if number of days is specified

                if match:
                    days = int(match.group(1))

                most_delayed_dev, delayed_prs = pr_analytics.most_delayed_developer(days)

                if isinstance(most_delayed_dev, str):  # Handle potential errors from most_delayed_developer

                    return most_delayed_dev

                else:

                    return f"The developer with the most delayed PRs in the last {days} days is: {most_delayed_dev}"

            elif "predict trend" in lower_prompt:
                match = re.search(r"(\d+)", prompt)
                if match:
                    future_weeks = int(match.group(1))
                    prediction = pr_analytics.predict_throughput_trend(future_weeks)
                    printk(prediction)
                    return f"Predicted PR throughput trend for the next {future_weeks} week(s): {prediction}"
                else:
                    # Default to one week prediction if no number is provided
                    prediction = pr_analytics.predict_throughput_trend(1)
                    return f"Predicted PR throughput trend for the next week: {prediction}"
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response['choices'][0]['message']['content']
        except Exception as e:
            logger.exception(f"Error querying OpenAI: {e}")
            return "An error occurred while processing your request."


# Flask Routes
@app.route("/")
def home():
    return " PR Analytics API is Running!"


@app.route("/average_merge_time", methods=["GET"])
def get_average_merge_time():
    days = request.args.get("days", default=30, type=int)
    return jsonify({"average_merge_time": pr_analytics.average_merge_time(days)})


@app.route("/bottlenecks", methods=["GET"])
def get_bottlenecks():
    return jsonify({"bottleneck_prs": pr_analytics.find_bottlenecks()})


@app.route("/throughput_trend", methods=["GET"])
def get_throughput_trend():
    weeks = request.args.get("weeks", default=3, type=int)
    result = pr_analytics.pr_throughput_trend(weeks)
    return jsonify({"pr_throughput_trend": result})


@app.route("/predict_merge_time", methods=["POST"])
def predict_merge_time():
    data = request.get_json()
    num_prs = data.get("num_prs", 1)
    prediction = pr_analytics.predict_merge_time(num_prs)
    return jsonify({"predicted_merge_time": prediction})


@app.route("/predict_bottlenecks", methods=["POST"])
def predict_bottlenecks():
    data = request.get_json()
    num_prs = data.get("num_prs", 1)
    prediction = pr_analytics.predict_bottlenecks(num_prs)
    return jsonify({"predicted_bottlenecks": prediction})


@app.route("/average_merge_time_last_month", methods=["POST"])
def get_average_merge_time_last_month():
    data = request.get_json()
    question = data.get("question", 30)
    response = ChatGPTHandler.query_chatgpt(question)
    return jsonify({"response": response})


@app.route("/predict_trend", methods=["POST"])
def predict_trend():
    try:
        data = request.get_json()
        future_weeks = data.get("future_weeks", 1)
        prediction = pr_analytics.predict_throughput_trend(future_weeks)
        return jsonify({"predicted_trend": prediction})
    except Exception as e:
        logger.exception(f"Error in /predict_trend route: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500


@app.route("/most_delayed_developer", methods=["POST"])
def get_most_delayed_developer():
    data = request.get_json()
    question = data.get("question", "")
    response = ChatGPTHandler.query_chatgpt(question)
    return jsonify({"response": response})


@app.route("/ask_gpt", methods=["POST"])
def ask_gpt():
    try:
        data = request.get_json()
        question = data.get("question", "")
        response = ChatGPTHandler.query_chatgpt(question)
        return jsonify({"response": response})
    except Exception as e:
        logger.exception(f"Error in /ask_gpt route: {e}")
        return jsonify({"error": "An unexpected error occurred."}), 500

# Route to serve the chatbot UI
@app.route("/chat")
def chat():
    return render_template("index.html")


# Run Flask App
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import io
import json
from datetime import datetime, timedelta
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from difflib import get_close_matches
import time


# Set page configuration
st.set_page_config(
    page_title="Health Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Keep only this CSS in your main set_page_config area:
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3366ff;
        text-align: center;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #4d88ff;
    }
    .metric-card {
        background-color: #f0f5ff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .good-health {
        color: #00cc66;
        font-weight: bold;
    }
    .moderate-health {
        color: #ffcc00;
        font-weight: bold;
    }
    .bad-health {
        color: #ff3333;
        font-weight: bold;
    }
    .stChat {
        border-radius: 10px;
        border: 1px solid #e6e6e6;
    }
        
    [data-theme="dark"] .recommendation-card {
        background-color: #333333 !important;
        color: white !important;
        border-left: 5px solid #9fd3c7 !important;
    }
    [data-theme="dark"] .recommendation-card * {
        color: inherit !important;
    }
            
    .recommendation-card {
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    border-left: 5px solid var(--primary-color);
    padding: 15px;
    margin: 10px 0;
    border-radius: 10px;
}
    
    /* Dark mode styles */
    [data-theme="dark"] {
    --primary-color: #9fd3c7;
    --background-color: #1a1a1a;
    --secondary-background-color: #2d2d2d;
    --text-color: #ffffff;
}

[data-theme="dark"] .main {
    background-color: var(--background-color);
    color: var(--text-color);
}

[data-theme="dark"] .sidebar .sidebar-content {
    background-color: var(--secondary-background-color);
    color: var(--text-color);

    }
    
    [data-theme="dark"] .metric-card {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid #444;
    }
    
    [data-theme="dark"] .recommendation-card {
        background-color: #333333;
        color: white;
        border-left: 5px solid #9fd3c7;
    }
    
    [data-theme="dark"] .stTextInput>div>div>input,
    [data-theme="dark"] .stSelectbox>div>div>select,
    [data-theme="dark"] .stTextArea>div>div>textarea {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }
    
    [data-theme="dark"] .stDataFrame {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }
    
    [data-theme="dark"] .st-bb {
        background-color: var(--secondary-background-color);
    }
    
    [data-theme="dark"] .stTable {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }
    
    [data-theme="dark"] .st-bw {
        background-color: var(--secondary-background-color);
    }
    
    [data-theme="dark"] .st-ax {
        background-color: var(--secondary-background-color);
    }
    
    [data-theme="dark"] .st-ay {
        background-color: var(--secondary-background-color);
    }
    
    [data-theme="dark"] .st-az {
        background-color: var(--secondary-background-color);
    }
    
    [data-theme="dark"] .st-b0 {
        background-color: var(--secondary-background-color);
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state variables if they don't exist
if 'health_data' not in st.session_state:
    st.session_state.health_data = pd.DataFrame(columns=[
        'Date', 'Steps', 'Distance_km', 'Calories_Burned', 'Heart_Rate', 
        'Weight_kg', 'Sleep_Hours', 'Water_Intake_Liters', 'Mood'
    ])

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False


# Function to greet user based on time
def greet_user():
    current_time = datetime.now()
    hour = current_time.hour
    
    if 5 <= hour < 12:
        greeting = "Good Morning"
    elif 12 <= hour < 17:
        greeting = "Good Afternoon"
    else:
        greeting = "Good Evening"
    
    return f"{greeting}, Welcome to your Personal Health Dashboard! "

# Function to convert between metrics
def calculate_health_metrics(data_point):
    """Calculate related health metrics based on available data"""
    updated_data = data_point.copy()
    
    # If steps provided but not distance
    if pd.notna(data_point.get('steps')) and pd.isna(data_point.get('distance_km')):
        updated_data['distance_km'] = round(data_point['steps'] * 0.0008, 2)  # Approx conversion
    
    # If distance provided but not steps
    if pd.notna(data_point.get('distance_km')) and pd.isna(data_point.get('steps')):
        updated_data['steps'] = int(data_point['distance_km'] / 0.0008)
    
    # Calculate calories if we have distance or steps but no calories
    if pd.isna(data_point.get('calories_burned')) and (pd.notna(data_point.get('steps')) or pd.notna(data_point.get('distance_km'))):
        weight = data_point.get('weight_kg', 70)  # Default weight if not provided
        if pd.notna(data_point.get('steps')):
            updated_data['calories_burned'] = int(0.04 * data_point['steps'] * (weight/70))
        else:
            updated_data['calories_burned'] = int(60 * data_point['distance_km'] * (weight/70))
    
    # Estimate heart rate if not provided
    if pd.isna(data_point.get('heart_rate')) and pd.notna(data_point.get('distance_km')):
        # Basic estimate - would be more accurate with age, fitness level etc.
        base_rate = 70  # Resting heart rate
        intensity_factor = min(data_point['distance_km'] / 5, 2)  # Limit the factor
        updated_data['heart_rate'] = int(base_rate + (70 * intensity_factor))
    
    return updated_data

# Function to analyze health status
def analyze_health_status(df):
    """Analyze health data and provide recommendations"""
    if df.empty:
        return "Not enough data to analyze health status", "not-available", "Please add health data to receive personalized recommendations.", ""
    
    # Calculate averages from last 7 days if available
    recent_data = df.sort_values(by='date', ascending=False).head(7)
    
    avg_steps = recent_data['steps'].mean()
    avg_sleep = recent_data['sleep_hours'].mean() if 'sleep_hours' in recent_data and not recent_data['sleep_hours'].isna().all() else 0
    avg_water = recent_data['water_intake_liters'].mean() if 'water_intake_liters' in recent_data and not recent_data['water_intake_liters'].isna().all() else 0
    avg_heart_rate = recent_data['heart_rate'].mean() 
    
    # Determine health status
    points = 0
    
    if avg_steps >= 10000:
        points += 2
    elif avg_steps >= 7000:
        points += 1
    
    if 6.5 <= avg_sleep <= 8.5:
        points += 2
    elif 5 <= avg_sleep < 6.5 or 8.5 < avg_sleep <= 10:
        points += 1
    
    if avg_water >= 2.5:
        points += 2
    elif avg_water >= 1.5:
        points += 1
    
    if 60 <= avg_heart_rate <= 80:
        points += 2
    elif 50 <= avg_heart_rate < 60 or 80 < avg_heart_rate <= 90:
        points += 1
    
    # Status based on points
    if points >= 6:
        status = "Good Health"
        status_class = "good-health"
    elif points >= 3:
        status = "Moderate Health"
        status_class = "moderate-health"
    else:
        status = "Needs Improvement"
        status_class = "bad-health"
    
    # Generate recommendations
    recommendations = []
    
    if avg_steps < 10000:
        recommendations.append("🚶‍♂️ **Activity:** Try to increase your daily steps to at least 10,000 for better cardiovascular health.")
    
    if avg_sleep < 6.5:
        recommendations.append("😴 **Sleep:** You may be sleep-deprived. Aim for 7-8 hours of quality sleep per night.")
    elif avg_sleep > 8.5:
        recommendations.append("😴 **Sleep:** Too much sleep can also be unhealthy. Try to maintain a consistent sleep schedule.")
    
    if avg_water < 2.5:
        recommendations.append("💧 **Hydration:** Increase your water intake to at least 2.5 liters per day for proper hydration.")
    
    if avg_heart_rate > 80:
        recommendations.append("❤️ **Heart Health:** Your average heart rate is elevated. Consider more relaxation techniques and consult a doctor if it persists.")
    
    if not recommendations:
        recommendations.append("🌟 **Maintenance:** You're doing great! Maintain your current healthy habits.")
    
    diet_plan = generate_diet_plan(status)
    
    return status, status_class, recommendations, diet_plan

# Function to generate diet plan based on health status
def generate_diet_plan(status):
    """Generate a personalized diet plan based on health status"""
    
    if status == "Good Health":
        return """
        **Balanced Diet Plan**
        
        *Breakfast:* Oatmeal with fruits and nuts or Greek yogurt with berries
        *Lunch:* Grilled chicken salad with olive oil dressing or Quinoa bowl with vegetables
        *Dinner:* Baked fish with roasted vegetables or Vegetable stir-fry with tofu
        *Snacks:* Fresh fruits, nuts, or vegetable sticks with hummus
        
        *Tip:* Maintain this balanced diet while ensuring adequate protein intake and plenty of vegetables.
        """
    
    elif status == "Moderate Health":
        return """
        **Improvement Diet Plan**
        
        *Breakfast:* Whole grain toast with avocado and eggs or Smoothie with spinach, banana, and protein
        *Lunch:* Lentil soup with a side salad or Turkey wrap with plenty of vegetables
        *Dinner:* Salmon with brown rice and steamed broccoli or Chicken and vegetable curry
        *Snacks:* Greek yogurt, apple with peanut butter, or a small handful of mixed nuts
        
        *Tip:* Focus on reducing processed foods and increasing fiber and protein intake.
        """
    
    else:  # Needs Improvement
        return """
        **Recovery Diet Plan**
        
        *Breakfast:* High-fiber cereal with milk and fruits or Vegetable omelet with whole grain toast
        *Lunch:* Large mixed vegetable salad with lean protein or Bean and vegetable soup
        *Dinner:* Grilled white fish with sweet potato and green vegetables or Tofu and vegetable stir-fry
        *Snacks:* Fresh fruits, vegetable sticks, or small portions of nuts
        
        *Tip:* Eliminate sugary drinks and processed foods. Increase water intake and focus on whole foods.
        """

# Function to get AI chatbot response
def get_chatbot_response(user_question, df):
    """Generate AI responses based on user questions and health data"""
    user_question = user_question.lower()
    
    if df.empty:
        return "I don't have enough data to answer your question. Please add some health data first."
    
    recent_data = df.sort_values(by='date', ascending=False)
    
    # Questions about steps
    if any(keyword in user_question for keyword in ['step', 'walk']):
        avg_steps = int(recent_data['steps'].mean())
        max_steps = int(recent_data['steps'].max())
        max_steps_date = recent_data.loc[recent_data['steps'].idxmax(), 'date'].strftime('%Y-%m-%d')
        
        return f"Your average daily steps are {avg_steps}. Your highest step count was {max_steps} on {max_steps_date}. The recommended goal is 10,000 steps per day for optimal health."
    
    # Questions about calories
    elif any(keyword in user_question for keyword in ['calorie', 'burn']):
        avg_calories = int(recent_data['calories_burned'].mean())
        weekly_calories = int(recent_data.head(7)['calories_burned'].sum())
        
        return f"You burn an average of {avg_calories} calories per day through your recorded activities. In the last week, you've burned approximately {weekly_calories} calories."
    
    # Questions about heart rate
    elif any(keyword in user_question for keyword in ['heart', 'pulse', 'bpm']):
        avg_heart_rate = int(recent_data['heart_rate'].mean())
        
        if avg_heart_rate < 60:
            status = "This is on the lower side which may be normal for athletes but could indicate bradycardia in others."
        elif avg_heart_rate <= 80:
            status = "This is within the healthy resting heart rate range."
        elif avg_heart_rate <= 100:
            status = "This is slightly elevated. Consider more relaxation techniques."
        else:
            status = "This is elevated and could indicate stress or other health concerns. Consider consulting a healthcare provider."
            
        return f"Your average heart rate is {avg_heart_rate} BPM. {status}"
    
    # Questions about sleep
    elif any(keyword in user_question for keyword in ['sleep', 'rest']):
        if 'sleep_hours' in recent_data.columns and not recent_data['sleep_hours'].isna().all():
            avg_sleep = round(recent_data['sleep_hours'].mean(), 1)
            
            if avg_sleep < 6:
                advice = "You may be sleep-deprived. Aim for 7-8 hours for better health outcomes."
            elif avg_sleep < 7:
                advice = "You're getting close to the recommended amount. Try to get a little more if possible."
            elif avg_sleep <= 9:
                advice = "You're getting a healthy amount of sleep. Keep it up!"
            else:
                advice = "You might be sleeping more than necessary. While occasional long sleep is fine, consistently sleeping too much could indicate health issues."
                
            return f"You sleep an average of {avg_sleep} hours per night. {advice}"
        else:
            return "I don't have any sleep data recorded. Please add sleep information to get insights."
    
    # Questions about water
    elif any(keyword in user_question for keyword in ['water', 'hydration', 'drink']):
        if 'water_intake_liters' in recent_data.columns and not recent_data['water_intake_liters'].isna().all():
            avg_water = round(recent_data['water_intake_liters'].mean(), 1)
            
            if avg_water < 1.5:
                advice = "This is below recommendations. Try to increase to at least 2-2.5 liters per day."
            elif avg_water < 2.5:
                advice = "You're doing okay, but could benefit from drinking a bit more."
            else:
                advice = "Great job staying hydrated!"
                
            return f"You drink an average of {avg_water} liters of water per day. {advice}"
        else:
            return "I don't have any water intake data recorded. Please add hydration information to get insights."
    
    # Questions about weight
    elif any(keyword in user_question for keyword in ['weight', 'kg', 'pound']):
        if 'weight_kg' in recent_data.columns and not recent_data['weight_kg'].isna().all():
            recent_weights = recent_data[~recent_data['weight_kg'].isna()].sort_values(by='date')
            if len(recent_weights) >= 2:
                first_weight = recent_weights.iloc[0]['weight_kg']
                last_weight = recent_weights.iloc[-1]['weight_kg']
                change = round(last_weight - first_weight, 1)
                
                if change > 0:
                    trend = f"You've gained {abs(change)} kg since {recent_weights.iloc[0]['date'].strftime('%Y-%m-%d')}."
                elif change < 0:
                    trend = f"You've lost {abs(change)} kg since {recent_weights.iloc[0]['date'].strftime('%Y-%m-%d')}."
                else:
                    trend = f"Your weight has remained stable since {recent_weights.iloc[0]['date'].strftime('%Y-%m-%d')}."
                
                return f"Your current weight is {last_weight} kg. {trend}"
            else:
                return f"Your current weight is {recent_weights.iloc[-1]['weight_kg']} kg. I need more data points to analyze trends."
        else:
            return "I don't have any weight data recorded. Please add weight information to get insights."
    
    # Questions about overall health
    elif any(keyword in user_question for keyword in ['health', 'overall', 'status', 'condition']):
        status, _, recommendations, _ = analyze_health_status(df)
        return f"Based on your data, your health status is: {status}. \n\nHere are some recommendations:\n\n" + "\n".join(recommendations)
    
    # General advice on improving health
    elif any(keyword in user_question for keyword in ['improve', 'better', 'advice', 'tip']):
        return """
        Here are some general health improvement tips:
        
        1. Aim for 10,000 steps per day
        2. Get 7-8 hours of quality sleep
        3. Drink at least 2.5 liters of water daily
        4. Incorporate strength training 2-3 times per week
        5. Practice mindfulness or meditation to reduce stress
        6. Eat a balanced diet with plenty of vegetables
        7. Limit processed foods and added sugars
        8. Take regular breaks when sitting for long periods
        
        Which area would you like more specific advice on?
        """
    
    # Diet related questions
    elif any(keyword in user_question for keyword in ['diet', 'eat', 'food', 'nutrition']):
        status, _, _, diet_plan = analyze_health_status(df)
        return f"Based on your health status ({status}), here's a recommended diet plan:\n\n{diet_plan}"
    
    # Fallback response
    else:
        return "I'm your health assistant! You can ask me about your steps, calories, heart rate, sleep, water intake, weight trends, or request advice on improving your health or diet."

# Function to save dataframe to various formats
def convert_df(df, file_format):
    """Convert DataFrame to the specified file format."""
    if file_format == 'csv':
        return df.to_csv(index=False).encode('utf-8')  # Ensure encoding for CSV
    elif file_format == 'json':
        return df.to_json(orient='records', indent=2).encode('utf-8')  # Pretty JSON
    elif file_format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Health_Data', index=False)
        return output.getvalue()  # Return binary data for Excel
    return None

# Function to generate different types of plots
def generate_plot(df, metric, title, plot_type='line', color='blue'):
    """Generate different types of plots for a given metric"""
    if df.empty or metric not in df.columns:
        st.warning(f"Cannot generate plot. The metric '{metric}' is missing or the data is empty.")
        return None

    # Ensure the 'date' column exists
    if 'date' not in df.columns:
        st.error("The 'date' column is missing from the data. Please ensure your data includes a 'date' column.")
        return None

    # Sort by date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    plot_df = df.sort_values(by='date')

    fig, ax = plt.subplots(figsize=(10, 5))

    if plot_type == 'line':
        ax.plot(plot_df['date'], plot_df[metric], marker='o', linestyle='-', color=color)
    elif plot_type == 'bar':
        ax.bar(plot_df['date'], plot_df[metric], color=color)
    elif plot_type == 'pie':
        value_counts = plot_df[metric].value_counts().head(5)
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    elif plot_type == 'hist':
        ax.hist(plot_df[metric], bins=10, color=color, edgecolor='black')

    ax.set_title(title, fontsize=14)
    if plot_type != 'pie':
        ax.set_xlabel('Date')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
def aggregate_weekly_data(df):
    """Aggregate data by week while preserving the date column for plotting"""
    if df.empty:
        return pd.DataFrame()
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create week column while keeping original date
    df['week_start'] = df['date'].dt.to_period('W').dt.start_time
    
    # Group by week and aggregate
    aggregated = df.groupby('week_start').agg({
        'steps': 'mean',
        'distance_km': 'mean',
        'calories_burned': 'mean',
        'heart_rate': 'mean',
        'weight_kg': 'mean',
        'sleep_hours': 'mean',
        'water_intake_liters': 'mean'
    }).reset_index()
    
    # Rename to 'date' for plotting compatibility
    aggregated = aggregated.rename(columns={'week_start': 'date'})
    
    return aggregated

def aggregate_monthly_data(df):
    """Aggregate data by month while preserving the date column for plotting"""
    if df.empty:
        return pd.DataFrame()
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create month column while keeping original date
    df['month_start'] = df['date'].dt.to_period('M').dt.start_time
    
    # Group by month and aggregate
    aggregated = df.groupby('month_start').agg({
        'steps': 'mean',
        'distance_km': 'mean',
        'calories_burned': 'mean',
        'heart_rate': 'mean',
        'weight_kg': 'mean',
        'sleep_hours': 'mean',
        'water_intake_liters': 'mean'
    }).reset_index()
    
    # Rename to 'date' for plotting compatibility
    aggregated = aggregated.rename(columns={'month_start': 'date'})
    
    return aggregated

# Function to display data analytics

def display_analytics():
    st.markdown("<h2 class='sub-header'>Analytics</h2>", unsafe_allow_html=True)

    if st.session_state.health_data.empty:
        st.info("No health data available. Please add data via the 'Upload Data' page.")
        return

    # Ensure column names are consistent and lowercase
    st.session_state.health_data.columns = st.session_state.health_data.columns.str.lower()

    # Check if date column exists, if not try common alternatives
    date_col = None
    for col in st.session_state.health_data.columns:
        if 'date' in col:
            date_col = col
            break
    
    if not date_col:
        st.error("No date column found in the data. Please ensure your data includes a date column.")
        return
    
    # Rename the date column to 'date' for consistency
    if date_col != 'date':
        st.session_state.health_data = st.session_state.health_data.rename(columns={date_col: 'date'})

    # Remove duplicate rows based on the 'date' column
    st.session_state.health_data = st.session_state.health_data.drop_duplicates(subset=['date'])

    # Convert the 'date' column to datetime format
    try:
        st.session_state.health_data['date'] = pd.to_datetime(st.session_state.health_data['date'], errors='coerce')
    except Exception as e:
        st.error(f"Error converting 'date' column to datetime: {str(e)}")
        return

    # Drop rows with invalid dates
    st.session_state.health_data = st.session_state.health_data.dropna(subset=['date'])

    # Sort by date
    st.session_state.health_data = st.session_state.health_data.sort_values('date')

    # Create tabs for different time periods
    tab1, tab2, tab3 = st.tabs(["Daily", "Weekly", "Monthly"])

    # Common plot type selection for all tabs
    plot_type_map = {
        "Line Chart": "line",
        "Bar Chart": "bar",
        "Pie Chart": "pie",
        "Histogram": "hist"
    }

    # Define metrics to display
    metrics = [
        ('steps', 'Steps', '#3366ff'),
        ('distance_km', 'Distance (km)', '#33cc33'),
        ('calories_burned', 'Calories (kcal)', '#ff3366'),
        ('heart_rate', 'Heart Rate (BPM)', '#ff6600'),
        ('sleep_hours', 'Sleep (hours)', '#9933ff'),
        ('water_intake_liters', 'Water (liters)', '#00ccff'),
        ('weight_kg', 'Weight (kg)', '#666666')
    ]

    # Filter out non-existent metrics
    available_metrics = [
        m for m in metrics if m[0] in st.session_state.health_data.columns and 
        not st.session_state.health_data[m[0]].isna().all()
    ]

    # Daily Tab
    with tab1:
        st.markdown("<h3>Daily Health Metrics</h3>", unsafe_allow_html=True)
        
        # Plot type selection with unique key
        plot_type = st.selectbox(
            "Select plot type:",
            ["Line Chart", "Bar Chart", "Pie Chart", "Histogram"],
            key="daily_plot_type"
        )
        
        # Create a 2-column layout for metrics
        col1, col2 = st.columns(2)
        
        # Plot metrics in alternating columns
        for i, (metric, title, color) in enumerate(available_metrics):
            with col1 if i % 2 == 0 else col2:
                fig = generate_plot(
                    st.session_state.health_data, 
                    metric, 
                    f'Daily {title}', 
                    plot_type=plot_type_map[plot_type],
                    color=color
                )
                if fig:
                    st.pyplot(fig)
         # Raw data table
        st.markdown("<h3>Raw Daily Data</h3>", unsafe_allow_html=True)
        st.dataframe(st.session_state.health_data.sort_values(by='date', ascending=False))
       

    # Weekly Tab
    with tab2:
        st.markdown("<h3>Weekly Health Metrics</h3>", unsafe_allow_html=True)
        
        # Aggregate weekly data
        weekly_data = aggregate_weekly_data(st.session_state.health_data)
        
        if not weekly_data.empty:
            # Plot type selection with unique key
            plot_type = st.selectbox(
                "Select plot type:",
                ["Line Chart", "Bar Chart", "Pie Chart", "Histogram"],
                key="weekly_plot_type"
            )
            
            # Create a 2-column layout for metrics
            col1, col2 = st.columns(2)
            
            # Plot metrics in alternating columns
            for i, (metric, title, color) in enumerate(available_metrics):
                with col1 if i % 2 == 0 else col2:
                    if metric in weekly_data.columns and not weekly_data[metric].isna().all():
                        fig = generate_plot(
                            weekly_data, 
                            metric, 
                            f'Weekly {title}', 
                            plot_type=plot_type_map[plot_type],
                            color=color
                        )
                        if fig:
                            st.pyplot(fig)
            #Raw data table
            st.markdown("<h3>Raw Weekly Data</h3>", unsafe_allow_html=True)
            st.dataframe(weekly_data.sort_values(by='date', ascending=False))                

    # Monthly Tab
    with tab3:
        st.markdown("<h3>Monthly Health Metrics</h3>", unsafe_allow_html=True)
        
        # Aggregate monthly data
        monthly_data = aggregate_monthly_data(st.session_state.health_data)
        
        if not monthly_data.empty:
            # Plot type selection with unique key
            plot_type = st.selectbox(
                "Select plot type:",
                ["Line Chart", "Bar Chart", "Pie Chart", "Histogram"],
                key="monthly_plot_type"
            )
            
            # Create a 2-column layout for metrics
            col1, col2 = st.columns(2)
            
            # Plot metrics in alternating columns
            for i, (metric, title, color) in enumerate(available_metrics):
                with col1 if i % 2 == 0 else col2:
                    if metric in monthly_data.columns and not monthly_data[metric].isna().all():
                        fig = generate_plot(
                            monthly_data, 
                            metric, 
                            f'Monthly {title}', 
                            plot_type=plot_type_map[plot_type],
                            color=color
                        )
                        if fig:
                            st.pyplot(fig)
            
            # Raw data table
            st.markdown("<h3>Raw Monthly Data</h3>", unsafe_allow_html=True)
            st.dataframe(monthly_data)

# Function to display health status
def display_health_status():
    st.markdown("<h2 class='sub-header'>Health Status & Recommendations</h2>", unsafe_allow_html=True)
    
    if st.session_state.health_data.empty:
        st.info("No health data available. Please add data via the 'Upload Data' page.")
        return
    
    try:
        status, status_class, recommendations, diet_plan = analyze_health_status(st.session_state.health_data)
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Display health status
            st.markdown(f"<h3>Current Health Status: <span class='{status_class}'>{status}</span></h3>", unsafe_allow_html=True)
            
            # Display recommendations
            st.markdown("<h4>Personalized Recommendations</h4>", unsafe_allow_html=True)
            for rec in recommendations:
                st.markdown(f"<div class='recommendation-card'>{rec}</div>", unsafe_allow_html=True)
            
            # Display diet plan
            st.markdown("<h4>Recommended Diet Plan</h4>", unsafe_allow_html=True)
            st.markdown(diet_plan)
        
        with col2:
            # Create a health status bar chart
            st.markdown("<h4>Health Status Breakdown</h4>", unsafe_allow_html=True)
            
            # Get recent data (last 7 days)
            recent_data = st.session_state.health_data.sort_values(by='date', ascending=False).head(7)
            
            # Calculate health metrics
            metrics = {
                'Steps': recent_data['steps'].mean() / 10000 * 100 if not recent_data['steps'].isna().all() else 0,
                'Sleep': recent_data['sleep_hours'].mean() / 8 * 100 if 'sleep_hours' in recent_data and not recent_data['sleep_hours'].isna().all() else 0,
                'Water': recent_data['water_intake_liters'].mean() / 2.5 * 100 if 'water_intake_liters' in recent_data and not recent_data['water_intake_liters'].isna().all() else 0,
                'Heart Rate': 100 if (60 <= recent_data['heart_rate'].mean() <= 80) else 70 if (50 <= recent_data['heart_rate'].mean() < 60 or 80 < recent_data['heart_rate'].mean() <= 90) else 40
            }
            
            # Create a DataFrame for plotting
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Score': list(metrics.values())
            })
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.barh(metrics_df['Metric'], metrics_df['Score'], color=['#3366ff', '#9933ff', '#00ccff', '#ff6600'])
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.0f}%', ha='left', va='center')
            
            # Set title and labels
            ax.set_title('Health Metrics Score (%)', fontsize=14)
            ax.set_xlim(0, 120)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            st.pyplot(fig)
            
            # Display key metrics
            st.markdown("<h4>Key Metrics (7-day average)</h4>", unsafe_allow_html=True)
            
            avg_steps = int(recent_data['steps'].mean()) if not recent_data['steps'].isna().all() else "N/A"
            avg_sleep = round(recent_data['sleep_hours'].mean(), 1) if 'sleep_hours' in recent_data and not recent_data['sleep_hours'].isna().all() else "N/A"
            avg_water = round(recent_data['water_intake_liters'].mean(), 1) if 'water_intake_liters' in recent_data and not recent_data['water_intake_liters'].isna().all() else "N/A"
            avg_hr = int(recent_data['heart_rate'].mean()) if not recent_data['heart_rate'].isna().all() else "N/A"
            
            # Display metrics
            st.metric("Average Steps", f"{avg_steps} / 10,000")
            st.metric("Average Sleep", f"{avg_sleep} hours / 7-8")
            st.metric("Average Water Intake", f"{avg_water} L / 2.5 L")
            st.metric("Average Heart Rate", f"{avg_hr} BPM / 60-80")

    except Exception as e:
        st.error(f"An error occurred while analyzing health status: {str(e)}")

# Function to display health report
def display_health_report():
    st.markdown("<h2 class='sub-header'>Health Report</h2>", unsafe_allow_html=True)
    
    if st.session_state.health_data.empty:
        st.info("No health data available. Please add data via the 'Upload Data' page.")
        return
    
    # Options for report period
    report_period = st.radio(
        "Select report period:",
        ["Last 7 days", "Last 30 days", "All data"],
        horizontal=True
    )
    
    # Filter data based on selected period
    today = datetime.today()
    if report_period == "Last 7 days":
        start_date = today - timedelta(days=7)
        period_name = "Last 7 days"
        filtered_df = st.session_state.health_data[pd.to_datetime(st.session_state.health_data['date']) >= start_date]
    elif report_period == "Last 30 days":
        start_date = today - timedelta(days=30)
        period_name = "Last 30 days"
        filtered_df = st.session_state.health_data[pd.to_datetime(st.session_state.health_data['date']) >= start_date]
    else:
        period_name = "All recorded data"
        filtered_df = st.session_state.health_data.copy()
    
    if filtered_df.empty:
        st.warning(f"No data available for the selected period: {period_name}")
        return
    
    # Display report
    st.markdown(f"## Health Report - {period_name}")
    
    # Summary statistics
    avg_steps = int(filtered_df['steps'].mean())
    avg_distance = round(filtered_df['distance_km'].mean(), 1)
    avg_calories = int(filtered_df['calories_burned'].mean())
    avg_heart = int(filtered_df['heart_rate'].mean())
    
    # Create metrics display
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg. Daily Steps", avg_steps, f"{int((avg_steps/10000)*100)}% of goal")
    col2.metric("Avg. Distance (km)", avg_distance)
    col3.metric("Avg. Calories", avg_calories)
    col4.metric("Avg. Heart Rate", avg_heart)

    # Add line and bar charts
    st.markdown("### Activity Trends")
    
    # Create tabs for different chart types
    tab1, tab2 = st.tabs(["Line Charts", "Bar Charts"])
    
    with tab1:
        # Steps trend
        fig_steps = generate_plot(filtered_df, 'steps', 'Steps Trend', 'line', '#3366ff')
        if fig_steps:
            st.pyplot(fig_steps)
        
        # Calories trend
        fig_cal = generate_plot(filtered_df, 'calories_burned', 'Calories Burned Trend', 'line', '#ff3366')
        if fig_cal:
            st.pyplot(fig_cal)
        
        # Heart rate trend
        fig_hr = generate_plot(filtered_df, 'heart_rate', 'Heart Rate Trend', 'line', '#ff6600')
        if fig_hr:
            st.pyplot(fig_hr)
    
    with tab2:
        # Steps trend
        fig_steps = generate_plot(filtered_df, 'steps', 'Steps Trend', 'bar', '#3366ff')
        if fig_steps:
            st.pyplot(fig_steps)
        
        # Calories trend
        fig_cal = generate_plot(filtered_df, 'calories_burned', 'Calories Burned Trend', 'bar', '#ff3366')
        if fig_cal:
            st.pyplot(fig_cal)
        
        # Heart rate trend
        fig_hr = generate_plot(filtered_df, 'heart_rate', 'Heart Rate Trend', 'bar', '#ff6600')
        if fig_hr:
            st.pyplot(fig_hr)
    
    
    # Export options
    st.markdown("### Export Report")
    export_format = st.selectbox("Select export format:", ['csv', 'json', 'excel'])
    
    if export_format == 'csv':
        csv = convert_df(filtered_df, 'csv')
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name=f"health_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
    elif export_format == 'json':
        json_data = convert_df(filtered_df, 'json')
        st.download_button(
            label="Download JSON Report",
            data=json_data,
            file_name=f"health_report_{datetime.now().strftime('%Y%m%d')}.json",
            mime='application/json'
        )
    elif export_format == 'excel':
        excel_data = convert_df(filtered_df, 'excel')
        st.download_button(
            label="Download Excel Report",
            data=excel_data,
            file_name=f"health_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

# Function to handle file upload
def display_file_upload():
    st.markdown("<h2 class='sub-header'>Upload Health Data</h2>", unsafe_allow_html=True)
    
    st.info("Upload your health data in CSV, Excel, or JSON format. The file should contain columns matching the health metrics.")
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'json'])
    
    if uploaded_file is not None:
        try:
            # Read file based on type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            # Debugging: Display the uploaded data
            st.write("Uploaded Data Preview:")
            st.dataframe(df.head())
            
            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Define required columns
            required_cols = ['date', 'steps', 'distance_km', 'calories_burned', 'heart_rate', 
                             'weight_kg', 'sleep_hours', 'water_intake_liters', 'mood']
            
            # Map similar column names
            column_mapping = {}
            for required_col in required_cols:
                similar_cols = get_close_matches(required_col, df.columns, n=1, cutoff=0.6)
                if similar_cols:
                    column_mapping[similar_cols[0]] = required_col
            
            # Rename columns based on mapping
            df = df.rename(columns=column_mapping)
            
            # Check for missing columns after mapping
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"The uploaded file is missing the following columns: {missing_cols}. Default values will be added.")
                for col in missing_cols:
                    # Add missing columns with default values (NaN)
                    df[col] = np.nan
            
            # Convert date column to datetime and remove timestamp
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
            else:
                st.error("The 'date' column is missing and is required for processing.")
                return
            
            # Drop rows with invalid dates
            df = df.dropna(subset=['date'])
            
            # Debugging: Display cleaned data
            st.write("Cleaned Data Preview:")
            st.dataframe(df.head())
            
            # Merge with existing data
            st.session_state.health_data = pd.concat([st.session_state.health_data, df], ignore_index=True).drop_duplicates(subset=['date'])
            
            # Debugging: Display merged data
            st.write("Merged Data Preview:")
            st.dataframe(st.session_state.health_data.head())
            
            st.success(f"Successfully uploaded {len(df)} records!")
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Sample data generator
    st.markdown("---")
    st.markdown("<h3>Generate Sample Data</h3>", unsafe_allow_html=True)
    st.warning("This will replace your current health data with sample data. Proceed with caution!")
    
    if st.button("Generate 30 Days of Sample Data"):
     with st.spinner("Generating sample data..."):
        # Generate date range for last 30 days
        dates = [(datetime.today() - timedelta(days=x)).date() for x in range(30, 0, -1)]
        
        # Generate random but somewhat realistic health data
        sample_data = []
        base_weight = random.uniform(60.0, 90.0)
        
        for i, date in enumerate(dates):
            # Base steps with some variation
            steps = random.randint(4000, 12000)
            distance_km = round(steps * 0.0008, 2)
            calories_burned = int(0.04 * steps * (base_weight / 70))
            heart_rate = random.randint(60, 80)
            weight_kg = round(base_weight + random.uniform(-1.5, 1.5), 1)
            sleep_hours = round(random.uniform(6.5, 8.5), 1)
            water_intake_liters = round(random.uniform(1.5, 3.5), 1)
            mood = random.choice(["Happy", "Neutral", "Sad", "Tired", "Excited"])
            
            sample_data.append({
                'date': date,
                'steps': steps,
                'distance_km': distance_km,
                'calories_burned': calories_burned,
                'heart_rate': heart_rate,
                'weight_kg': weight_kg,
                'sleep_hours': sleep_hours,
                'water_intake_liters': water_intake_liters,
                'mood': mood
            })
        
        # Create DataFrame and store in session state
        st.session_state.health_data = pd.DataFrame(sample_data)
        
        st.success("Generated 30 days of sample health data!")
        st.dataframe(st.session_state.health_data)
# Function to display health assistant
def display_health_assistant():
    st.markdown("<h2 class='sub-header'>Health Assistant</h2>", unsafe_allow_html=True)
    
    st.info("Ask me any questions about your health data, or for advice on improving your fitness, diet, and wellbeing.")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Ask me about your health data..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = get_chatbot_response(prompt, st.session_state.health_data)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Function to display dashboard
def display_dashboard():
    st.markdown("<h2 class='sub-header'>Dashboard Overview</h2>", unsafe_allow_html=True)
    
    if st.session_state.health_data.empty:
        st.info("No health data available. Please add data via the 'Upload Data' page.")
        return
    
    # Ensure date column is datetime
    st.session_state.health_data['date'] = pd.to_datetime(st.session_state.health_data['date'])
    
    # Get most recent data
    recent_data = st.session_state.health_data.sort_values(by='date', ascending=False).head(1).iloc[0]
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display recent metrics
        st.markdown("<h3>Recent Health Metrics</h3>", unsafe_allow_html=True)
        
        # Create metrics cards
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Steps", f"{int(recent_data['steps']):,}" if pd.notna(recent_data['steps']) else "N/A", "10,000 goal")
            st.metric("Distance", f"{recent_data['distance_km']:.1f} km" if pd.notna(recent_data['distance_km']) else "N/A", "8 km goal")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Calories", f"{int(recent_data['calories_burned']):,}" if pd.notna(recent_data['calories_burned']) else "N/A", "500 kcal goal")
            st.metric("Heart Rate", f"{int(recent_data['heart_rate'])} BPM" if pd.notna(recent_data['heart_rate']) else "N/A", "60-80 BPM")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with metrics_col2:
            if 'sleep_hours' in recent_data and pd.notna(recent_data['sleep_hours']):
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Sleep", f"{recent_data['sleep_hours']:.1f} hours", "7-8 hours")
                st.metric("Water", f"{recent_data['water_intake_liters']:.1f} L" if 'water_intake_liters' in recent_data and pd.notna(recent_data['water_intake_liters']) else "N/A", "2.5 L goal")
                st.markdown("</div>", unsafe_allow_html=True)
            
            if 'weight_kg' in recent_data and pd.notna(recent_data['weight_kg']):
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Weight", f"{recent_data['weight_kg']:.1f} kg")
                st.metric("Mood", recent_data['mood'] if 'mood' in recent_data and pd.notna(recent_data['mood']) else "N/A")
                st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
    
    
    # Get last 7 days of data
     recent_7_days = st.session_state.health_data.sort_values(by='date', ascending=False).head(7)
    
    if not recent_7_days.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot steps
        ax.plot(recent_7_days['date'], recent_7_days['steps'], label='Steps', color='#3366ff', marker='o')
        ax.set_ylabel('Steps', color='#3366ff')
        ax.tick_params(axis='y', labelcolor='#3366ff')
        
        # Create second y-axis for distance
        ax2 = ax.twinx()
        ax2.plot(recent_7_days['date'], recent_7_days['distance_km'], label='Distance (km)', color='#33cc33', marker='s', linestyle='--')
        ax2.set_ylabel('Distance (km)', color='#33cc33')
        ax2.tick_params(axis='y', labelcolor='#33cc33')
        
        # Format x-axis
        ax.set_xlabel('Date')
        plt.xticks(rotation=45)
        
        # Add legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
        
        # Add grid and style
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Not enough data to display recent activity chart.")

    # Initialize session state for dark mode if it doesn't exist
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False 

 

#simplified toggle function

def toggle_dark_mode():
    """Toggle dark mode state without directly calling rerun"""
    st.session_state.dark_mode = not st.session_state.dark_mode
   
    

    # Set the theme in Streamlit's config
    if st.session_state.dark_mode:
        st._config.set_option("theme.base", "dark")
    else:
        st._config.set_option("theme.base", "light")
    

# Main app function
def main():
    # Check if we need to rerun due to dark mode change
    if st.session_state.get('need_rerun', False):
        st.session_state.need_rerun = False
        
    
    # Dark mode toggle button in sidebar
    with st.sidebar:
        st.button(
            "🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode",
            key="dark_mode_toggle",
            on_click=toggle_dark_mode
        )
    
    # Apply dark mode by setting the theme attribute on the HTML element
    st.markdown(
        f'<html data-theme="{"dark" if st.session_state.dark_mode else "light"}"></html>',
        unsafe_allow_html=True
    )
    
    # Rest of your app code...
    st.markdown(f"<h1 class='main-header'>{greet_user()}</h1>", unsafe_allow_html=True)
    
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Go to",
        ["Upload Data", "Dashboard", "Health Status", "Health Report", "Analytics", "Health Assistant"]
    )
    
    # Display the appropriate page based on selection
    if app_mode == "Upload Data":
        display_file_upload()
    elif app_mode == "Dashboard":
        display_dashboard()
    elif app_mode == "Health Status":
        display_health_status()
    elif app_mode == "Health Report":
        display_health_report()
    elif app_mode == "Analytics":
        display_analytics()
    elif app_mode == "Health Assistant":
        display_health_assistant()
    
    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
    Personal Health Dashboard v1.0  
    Track and analyze your health metrics.  
    [GitHub Repository](#)  
    """)

if __name__ == "__main__":
    main()

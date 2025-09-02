from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the machine learning pipeline from the pickle file
pipe = pickle.load(open('pipe.pkl', 'rb'))

# List of teams for the dropdown menus
teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]

# List of cities for the dropdown menu
cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 
    'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 
    'Christchurch', 'Trinidad'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route for the web application. Handles both GET requests (displaying the form)
    and POST requests (processing form data and making a prediction).
    """
    prediction = None
    
    # This block executes when the user submits the form
    if request.method == 'POST':
        try:
            # --- 1. Get User Input from the Form ---
            batting_team = request.form['batting_team']
            bowling_team = request.form['bowling_team']
            city = request.form['city']
            
            # --- NEW: Check if number fields are empty before converting ---
            score_input = request.form['current_score']
            overs_input = request.form['overs']
            wickets_input = request.form['wickets']
            last_five_input = request.form['last_five']

            if not all([score_input, overs_input, wickets_input, last_five_input]):
                 # If any field is empty, show a user-friendly error
                prediction = "Error: Please fill out all the score and overs fields."
                return render_template('index.html', teams=sorted(teams), cities=sorted(cities), prediction=prediction)

            # Convert to numbers now that we know they are not empty
            current_score = int(score_input)
            overs = float(overs_input)
            wickets = int(wickets_input)
            last_five = int(last_five_input)


            # --- 2. Calculate Derived Features ---
            # These are the features the model needs but aren't directly input by the user.
            
            balls_left = 120 - (overs * 6)
            wickets_left = 10 - wickets
            
            # To avoid a ZeroDivisionError if overs is 0
            if overs == 0:
                crr = 0.0
            else:
                crr = current_score / overs

            # --- 3. Create a DataFrame for the Model ---
            # The column names here MUST EXACTLY match the names used when training the model.
            
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'current_score': [current_score],
                'balls_left': [balls_left],
                'wicket_left': [wickets_left],        # CORRECTED: Name is 'wicket_left' (singular)
                'current_run_rate': [crr],
                'last_five': [last_five]              # CORRECTED: Name is 'last_five'
            })

            # --- 4. Make the Prediction ---
            result = pipe.predict(input_df)
            # The result is a numpy array, so we extract the first element and convert to integer
            prediction = int(result[0])

        except Exception as e:
            # If any other error occurs, show a generic error message
            prediction = f"Error: {e}. Please check all your inputs are valid."

    # --- 5. Render the Page ---
    # This will render the page on the initial GET request and also after a POST request
    return render_template('index.html', teams=sorted(teams), cities=sorted(cities), prediction=prediction)

if __name__ == '__main__':
    # Run the Flask app in debug mode for development
    app.run(debug=True)

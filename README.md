# Football Player Transfer Value Prediction

This project analyzes football player transfer values using data-driven insights and machine learning models.

## Insights from the Data

### Q1: How Do Transfer Values Compare Across Positions?
- Transfer values vary significantly by position, with attacking players generally commanding higher fees.

### Q2: What’s the Relationship Between Age and Transfer Value?
- Younger players with high potential tend to have higher transfer values, while older players often see a decline.

### Q3: How Does Having Awards Affect Transfer Value?
- Players with major awards tend to have significantly higher transfer values due to increased market demand.

### Q4: Which Positions Have the Most Injured Players?
- Defensive players tend to suffer more injuries compared to other positions, potentially affecting their market value.

### Q5: How Much Value Do Players Retain Over Time?
- Some players experience a significant drop from their peak value to their current value, often due to injuries or a decline in performance. On the other hand, some players maintain a relatively stable value over time, suggesting consistency in performance and fewer injuries or setbacks.

## Model Training & Performance

Each position has its own linear regression model for better accuracy. This approach was chosen because different positions have unique characteristics that influence transfer values. Factors such as goalkeeping performance, defensive reliability, midfield creativity, and attacking efficiency vary significantly, requiring separate models to capture these nuances effectively.

### Goalkeeper (GK) Model:
- **Mean Absolute Error (MAE) - Training:** 1,033,843
- **Mean Absolute Error (MAE) - Testing:** 1,185,792
- **R² Score - Training:** 0.7931
- **R² Score - Testing:** 0.6984

### Defender (DEF) Model:
- **Mean Absolute Error (MAE) - Training:** 1,791,897
- **Mean Absolute Error (MAE) - Testing:** 1,886,594
- **R² Score - Training:** 0.7849
- **R² Score - Testing:** 0.7412

### Midfielder (MID) Model:
- **Mean Absolute Error (MAE) - Training:** 2,163,158
- **Mean Absolute Error (MAE) - Testing:** 2,446,320
- **R² Score - Training:** 0.7698
- **R² Score - Testing:** 0.7533

### Attacker (ATT) Model:
- **Mean Absolute Error (MAE) - Training:** 2,600,876
- **Mean Absolute Error (MAE) - Testing:** 2,475,373
- **R² Score - Training:** 0.7788
- **R² Score - Testing:** 0.6263

## Try the App
[Click here to access the app](https://football-player-price-predictor.streamlit.app)  <!-- Replace # with your app link -->

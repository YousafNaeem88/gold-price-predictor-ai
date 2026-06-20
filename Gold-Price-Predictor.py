# ============================================================
#  GOLD PRICE PREDICTOR - AI Project
#  Uses Linear Regression to predict future gold prices
#  Easy to understand for beginners!
# ============================================================

# STEP 1: Import the tools (libraries) we need
import numpy as np                          # For math/numbers
import pandas as pd                         # For handling data tables
import matplotlib.pyplot as plt             # For drawing graphs
from sklearn.linear_model import LinearRegression  # The AI model
from sklearn.model_selection import train_test_split  # To split data
from sklearn.metrics import mean_absolute_error, r2_score  # To measure accuracy

print("=" * 55)
print("       GOLD PRICE PREDICTOR - AI Project")
print("=" * 55)

# ============================================================
# STEP 2: Create Sample Gold Price Data
# (In a real project, you'd download this from the internet)
# Prices are in USD per ounce, from 2015 to 2024
# ============================================================

data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Gold_Price': [1060, 1145, 1257, 1268, 1477, 1770, 1798, 1800, 1943, 2300]
}

# Convert to a DataFrame (like a spreadsheet in Python)
df = pd.DataFrame(data)

print("\n📋 Gold Price Data (2015-2024):")
print("-" * 35)
for _, row in df.iterrows():
    bar = "█" * int(row['Gold_Price'] / 100)
    print(f"  {int(row['Year'])}: ${int(row['Gold_Price']):,}  {bar}")

# ============================================================
# STEP 3: Prepare the data for the AI
# X = Input  (the year)
# y = Output (the gold price we want to predict)
# ============================================================

X = df[['Year']]        # Input feature (2D array needed)
y = df['Gold_Price']    # Target value to predict

# Split data: 80% for training, 20% for testing
# This is like studying (training) and then taking an exam (testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📚 Training samples : {len(X_train)}")
print(f"🧪 Testing samples  : {len(X_test)}")

# ============================================================
# STEP 4: Train the AI Model (Linear Regression)
# The model learns the pattern: as year goes up, price goes up
# ============================================================

model = LinearRegression()   # Create the model
model.fit(X_train, y_train)  # Train it on our data

print("\n🤖 Model Training Complete!")
print(f"   Formula: Price = {model.coef_[0]:.2f} × Year + ({model.intercept_:.2f})")
print(f"   This means: every year, gold goes up ~${model.coef_[0]:.0f}")

# ============================================================
# STEP 5: Test the model - see how accurate it is
# ============================================================

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2  = r2_score(y_test, predictions)

print("\n📊 Model Accuracy:")
print(f"   R² Score          : {r2:.4f}  (1.0 = perfect)")
print(f"   Mean Abs Error    : ${mae:.2f} per ounce")
if r2 > 0.9:
    print("   ✅ Model is VERY GOOD!")
elif r2 > 0.7:
    print("   ✅ Model is GOOD!")
else:
    print("   ⚠️  Model needs more data")

# ============================================================
# STEP 6: Make Future Predictions
# ============================================================

future_years = np.array([[2025], [2026], [2027], [2028], [2030]])
future_prices = model.predict(future_years)

print("\n🔮 Future Gold Price Predictions:")
print("-" * 35)
for year, price in zip(future_years.flatten(), future_prices):
    print(f"   {year}: ${price:,.0f} per ounce")

# ============================================================
# STEP 7: Draw a Nice Graph
# ============================================================

plt.figure(figsize=(12, 6))
plt.style.use('seaborn-v0_8-darkgrid')

# Plot actual historical prices
plt.plot(df['Year'], df['Gold_Price'],
         'o-', color='gold', linewidth=2.5,
         markersize=8, label='📈 Actual Prices', zorder=3)

# Plot the model's trend line (extended to future)
all_years = np.arange(2015, 2031).reshape(-1, 1)
all_predictions = model.predict(all_years)
plt.plot(all_years, all_predictions,
         '--', color='orange', linewidth=2,
         alpha=0.7, label='🤖 AI Prediction Line')

# Highlight future predictions
plt.scatter(future_years, future_prices,
            color='red', s=120, zorder=4,
            label='🔮 Future Predictions')

# Labels and formatting
plt.title('🥇 Gold Price Predictor - AI Linear Regression Model',
          fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Gold Price (USD per ounce)', fontsize=12)
plt.legend(fontsize=11)
plt.xticks(range(2015, 2031, 1), rotation=45)
plt.tight_layout()

# Save the graph as an image
plt.savefig('gold_price_chart.png', dpi=150, bbox_inches='tight')
print("\n📸 Graph saved as: gold_price_chart.png")

plt.show()

# ============================================================
# STEP 8: Summary for your teacher
# ============================================================

print("\n" + "=" * 55)
print("           PROJECT SUMMARY")
print("=" * 55)
print("""
 This project uses AI (Machine Learning) to predict
 how much gold will cost in the future.

 HOW IT WORKS (Simple Explanation):
 ┌─────────────────────────────────────────────┐
 │  1. COLLECT DATA   → Past gold prices       │
 │  2. TRAIN MODEL    → AI learns the pattern  │
 │  3. TEST MODEL     → Check accuracy         │
 │  4. PREDICT        → Guess future prices    │
 └─────────────────────────────────────────────┘

 ALGORITHM USED: Linear Regression
 → Finds the best straight line through the data
 → Uses that line to predict future values

 LIBRARIES USED:
  • numpy     - Math operations
  • pandas    - Data handling
  • sklearn   - Machine Learning model
  • matplotlib - Drawing graphs
""")
print("=" * 55)
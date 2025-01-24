import streamlit as st

# Title of the app
st.title("Simple Streamlit App")

# Text input
user_input = st.text_input("Enter some text:", "Hello, Streamlit!")

# Display the input back to the user
st.write(f"You entered: {user_input}")

# Button example
if st.button("Click Me"):
    st.write("Button clicked!")

# Example plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [10, 20, 25, 30])
ax.set_title("Example Plot")
st.pyplot(fig)

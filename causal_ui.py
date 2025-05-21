import streamlit as st
import matplotlib.pyplot as plt
from psm_inference import run_psm_inference
from dml_inference import run_dml_inference

# streamlit UI configuration
st.set_page_config(page_title="Causal Inference Demonstration", layout="centered")

if 'page' not in st.session_state:
    st.session_state.page = 'home'

def main():
    if st.session_state.page == 'home':
        st.title("Welcome to Causal Inference Demonstration")
        st.write("Which approach do you want?")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("PSM Inference"):
                st.session_state.page = 'psm'
        with col2:
            if st.button("DML Inference"):
                st.session_state.page = 'dml'
        with col3:
            if st.button("Exit"):
                st.stop()
    
    elif st.session_state.page == 'psm':
        st.header("PSM Inference Results")
        try:
            fig, output = run_psm_inference()
            st.pyplot(fig)
            st.text(output)
            st.markdown("""
                **What’s Happening?**  
                For the Lalonde dataset, we’re studying the effect of a job training program (the treatment) on earnings in 1978. The treatment group received training, while the control group did not. We accounted for confounding variables like age, education, race, marital status, and prior earnings (1974 and 1975), which might affect both training and earnings. Using Propensity Score Matching (PSM), we paired similar individuals from both groups based on these variables. The plot shows how well the groups match before and after PSM. The Average Treatment Effect (ATE) tells us the average difference in earnings due to the training program.
            """)
            plt.close(fig)  # Close figure to free memory
            if st.button("Back to Home"):
                st.session_state.page = 'home'
        except Exception as e:
            st.error(f"Error running PSM inference: {str(e)}")
    
    elif st.session_state.page == 'dml':
        st.header("DML Inference Results")
        try:
            fig, output = run_dml_inference()
            st.pyplot(fig)
            st.text(output)
            st.markdown("""
                **What’s happening?**  
                For the dataset, we’re examining how being in the UK (the treatment) affects total purchase amounts. The treatment group consists of UK customers, while the control group includes non-UK customers. We considered confounding variables like purchase quantity, price, and customer ID, which could influence both location and spending. Using Double Machine Learning (DML), we estimate the treatment’s impact while controlling for these variables. The plot shows the distribution of individual treatment effects (CATE) for both groups, and the Average Treatment Effect (ATE) indicates the average difference in purchase amounts due to being in the UK.
            """)
            plt.close(fig)  # Close figure to free memory
            if st.button("Back to Home"):
                st.session_state.page = 'home'
        except Exception as e:
            st.error(f"Error running DML inference: {str(e)}")

if __name__ == "__main__":
    main()
# Data-Science-Interview-Assistant

1. Problem: LLMs hallucinate on domain-specific Q&A
2. Approach: Fine-tune flan-t5-base on 683 DS interview Q&A pairs
3. Result: 45% ROUGE improvement over baseline, PEFT matching 
           full fine-tune at 1% of parameters
4. Alignment: PPO quality tuning explored, empirically documented
              diminishing returns on strong PEFT baseline
5. Deployment: RAG-powered Streamlit app grounding answers 
               in retrieved Q&A context

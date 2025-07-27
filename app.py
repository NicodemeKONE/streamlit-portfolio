import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Portfolio - Nicodème Koné",
    page_icon="📊",
    layout="wide"
)

def send_email(name, email, subject, message):
    """Fonction pour envoyer un email"""
    try:
        # Configuration SMTP depuis les secrets
        smtp_server = st.secrets["email"]["smtp_server"]
        smtp_port = st.secrets["email"]["smtp_port"]
        sender_email = st.secrets["email"]["sender_email"]
        sender_password = st.secrets["email"]["sender_password"]
        receiver_email = st.secrets["email"]["receiver_email"]
        
        # Créer le message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = f"Portfolio Contact: {subject}"
        
        # Corps du message
        email_body = f"""
        Nouveau message depuis votre portfolio :
        
        Nom: {name}
        Email: {email}
        Sujet: {subject}
        
        Message:
        {message}
        
        ---
        Envoyé le {datetime.now().strftime('%d/%m/%Y à %H:%M')}
        """
        
        msg.attach(MIMEText(email_body, 'plain'))
        
        # Envoyer l'email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'envoi: {str(e)}")
        return False

# Interface utilisateur
st.title("📊 Portfolio - Data Scientist")
st.markdown("---")

# Section présentation
st.markdown("""
## 👋 Bienvenue sur mon portfolio

Spécialisé en data science et analyse de données, je transforme vos données en insights actionnables.
""")

# Section contact
st.markdown("## 📧 Contactez-moi")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("""
    ### 🤝 Travaillons ensemble
    
    Que vous portiez un projet de data science, ayez besoin d'une analyse rigoureuse 
    ou souhaitiez concevoir des visualisations claires et percutantes, je suis à votre écoute.
    
    **📍 Localisation:** Caen, Normandie, France  
    **📧 Email:** nicoetude@email.com  
    **📱 Téléphone:** +33 7 58 55 30 80  
    **💼 LinkedIn:** [linkedin.com/in/nicodeme-kone](https://www.linkedin.com/in/nicodeme-kone/)
    """)

with col2:
    st.markdown("### 📝 Formulaire de contact")
    
    with st.form("contact_form"):
        name = st.text_input("Nom complet *")
        email = st.text_input("Adresse email *")
        subject = st.text_input("Sujet *")
        message = st.text_area("Message *", height=150)
        
        submitted = st.form_submit_button("📧 Envoyer le message")
        
        if submitted:
            if name and email and subject and message:
                with st.spinner("Envoi en cours..."):
                    if send_email(name, email, subject, message):
                        st.success("✅ Message envoyé avec succès !")
                        st.balloons()
                    else:
                        st.error("❌ Erreur lors de l'envoi.")
            else:
                st.error("⚠️ Veuillez remplir tous les champs.")
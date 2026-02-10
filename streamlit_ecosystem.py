# streamlit_ecosystem.py - Version Streamlit corrigée
# Intégration complète pour portfolio GitHub/Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
import json
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="🌍 Simulateur d'Écosystème",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un meilleur design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E7D32;
        margin-bottom: 2rem;
    }
    .ecosystem-stats {
        background: linear-gradient(90deg, #e3f2fd, #f3e5f5);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .animal-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: bold;
    }
    .status-running {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-paused {
        color: #FF9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Classes simplifiées pour Streamlit
class StreamlitAnimal:
    def __init__(self, species, x, y, life, max_life, age=0):
        self.species = species
        self.x = x
        self.y = y
        self.life = life
        self.max_life = max_life
        self.age = age
        self.direction_x = random.choice([-1, 0, 1])
        self.direction_y = random.choice([-1, 0, 1])
        
    def move(self, grid_width, grid_height):
        """Déplacement aléatoire"""
        new_x = max(0, min(grid_width - 1, self.x + self.direction_x))
        new_y = max(0, min(grid_height - 1, self.y + self.direction_y))
        
        # Changer de direction parfois
        if random.random() < 0.3:
            self.direction_x = random.choice([-1, 0, 1])
            self.direction_y = random.choice([-1, 0, 1])
            
        self.x, self.y = new_x, new_y
        
    def lose_energy(self, amount=1):
        """Perte d'énergie"""
        self.life = max(0, self.life - amount)
        
    def gain_energy(self, amount=5):
        """Gain d'énergie"""
        self.life = min(self.max_life, self.life + amount)
        
    def is_alive(self):
        return self.life > 0
        
    def age_one_year(self):
        self.age += 1

class EcosystemSimulator:
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.animals = []
        self.resources = []
        self.turn = 0
        self.stats_history = []
        
        # Configuration
        self.animal_configs = {
            'Mouse': {'emoji': '🐭', 'color': '#8D6E63', 'max_life': 15, 'energy_cost': 1},
            'Cow': {'emoji': '🐄', 'color': '#6D4C41', 'max_life': 40, 'energy_cost': 1},
            'Lion': {'emoji': '🦁', 'color': '#FF8F00', 'max_life': 50, 'energy_cost': 2},
            'Dragon': {'emoji': '🐲', 'color': '#D32F2F', 'max_life': 100, 'energy_cost': 1}
        }
        
    def add_animal(self, species, count):
        """Ajoute des animaux à l'écosystème"""
        config = self.animal_configs[species]
        for _ in range(count):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            life = random.randint(config['max_life'] // 2, config['max_life'])
            animal = StreamlitAnimal(species, x, y, life, config['max_life'])
            self.animals.append(animal)
            
    def add_resources(self, herb_count, water_count):
        """Ajoute des ressources"""
        self.resources = []
        
        # Herbes
        for _ in range(herb_count):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.resources.append({'type': 'Herb', 'x': x, 'y': y, 'emoji': '🌿', 'value': 5})
            
        # Eau
        for _ in range(water_count):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.resources.append({'type': 'Water', 'x': x, 'y': y, 'emoji': '💧', 'value': 10})
            
    def simulate_turn(self):
        """Simule un tour"""
        self.turn += 1
        
        # Déplacer les animaux
        for animal in self.animals:
            if animal.is_alive():
                animal.move(self.width, self.height)
                animal.lose_energy(self.animal_configs[animal.species]['energy_cost'])
                
                # Consommer ressources
                for resource in self.resources[:]:
                    if animal.x == resource['x'] and animal.y == resource['y']:
                        animal.gain_energy(resource['value'])
                        self.resources.remove(resource)
                        break
        
        # Vieillissement (tous les 10 tours)
        if self.turn % 10 == 0:
            for animal in self.animals:
                animal.age_one_year()
                # Mort de vieillesse
                if animal.age > 20 and random.random() < 0.1:
                    animal.life = 0
        
        # Reproduction simple
        if self.turn % 15 == 0:
            self.simple_reproduction()
            
        # Prédation simple
        self.simple_predation()
        
        # Nettoyer les morts
        self.animals = [a for a in self.animals if a.is_alive()]
        
        # Respawn des ressources
        if self.turn % 5 == 0:
            self.respawn_resources()
            
        # Sauvegarder les stats
        self.save_stats()
        
    def simple_predation(self):
        """Prédation simplifiée"""
        predation_rules = {
            'Dragon': ['Lion', 'Cow', 'Mouse'],
            'Lion': ['Cow', 'Mouse']
        }
        
        for predator in self.animals:
            if not predator.is_alive():
                continue
                
            if predator.species in predation_rules:
                for prey in self.animals:
                    if (prey.species in predation_rules[predator.species] and 
                        prey.is_alive() and
                        abs(predator.x - prey.x) <= 1 and 
                        abs(predator.y - prey.y) <= 1):
                        
                        # Chasse réussie
                        if random.random() < 0.3:  # 30% de chance
                            prey.life = 0
                            predator.gain_energy(20)
                            break
                            
    def simple_reproduction(self):
        """Reproduction simplifiée"""
        species_groups = {}
        
        # Grouper par espèce
        for animal in self.animals:
            if animal.is_alive() and animal.life > animal.max_life * 0.7:
                if animal.species not in species_groups:
                    species_groups[animal.species] = []
                species_groups[animal.species].append(animal)
        
        # Reproduction pour chaque espèce
        for species, animals in species_groups.items():
            if len(animals) >= 2:
                parent1, parent2 = random.sample(animals, 2)
                
                # Créer un bébé
                if random.random() < 0.2:  # 20% de chance
                    config = self.animal_configs[species]
                    baby = StreamlitAnimal(
                        species, 
                        parent1.x, 
                        parent1.y, 
                        config['max_life'] // 2,
                        config['max_life']
                    )
                    self.animals.append(baby)
                    
                    # Coût énergétique pour les parents
                    parent1.lose_energy(15)
                    parent2.lose_energy(15)
                    
    def respawn_resources(self):
        """Fait réapparaître des ressources"""
        current_herbs = len([r for r in self.resources if r['type'] == 'Herb'])
        current_water = len([r for r in self.resources if r['type'] == 'Water'])
        
        # Maintenir un nombre minimum
        min_herbs = 15
        min_water = 8
        
        if current_herbs < min_herbs:
            for _ in range(min_herbs - current_herbs):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                self.resources.append({'type': 'Herb', 'x': x, 'y': y, 'emoji': '🌿', 'value': 5})
                
        if current_water < min_water:
            for _ in range(min_water - current_water):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                self.resources.append({'type': 'Water', 'x': x, 'y': y, 'emoji': '💧', 'value': 10})
                
    def save_stats(self):
        """Sauvegarde les statistiques"""
        stats = {'turn': self.turn}
        
        # Compter par espèce
        for species in self.animal_configs.keys():
            count = len([a for a in self.animals if a.species == species and a.is_alive()])
            stats[species] = count
            
        # Ressources
        stats['Herbs'] = len([r for r in self.resources if r['type'] == 'Herb'])
        stats['Water'] = len([r for r in self.resources if r['type'] == 'Water'])
        stats['Total_Animals'] = len(self.animals)
        
        self.stats_history.append(stats)
        
    def get_grid_data(self):
        """Retourne les données pour affichage en grille"""
        grid = np.zeros((self.height, self.width), dtype=object)
        
        # Placer les ressources
        for resource in self.resources:
            if 0 <= resource['y'] < self.height and 0 <= resource['x'] < self.width:
                grid[resource['y']][resource['x']] = resource['emoji']
        
        # Placer les animaux (par-dessus les ressources)
        for animal in self.animals:
            if animal.is_alive() and 0 <= animal.y < self.height and 0 <= animal.x < self.width:
                emoji = self.animal_configs[animal.species]['emoji']
                grid[animal.y][animal.x] = emoji
                
        return grid

# Interface Streamlit
def main():
    # Titre principal
    st.markdown('<h1 class="main-header">🌍 Simulateur d\'Écosystème Interactif</h1>', 
                unsafe_allow_html=True)
    
    # Initialisation de l'état de session
    if 'ecosystem' not in st.session_state:
        st.session_state.ecosystem = None
        st.session_state.running = False
        st.session_state.auto_speed = 1.0
        st.session_state.last_update = time.time()
        
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Taille de grille
        st.subheader("🌍 Taille de la Planète")
        grid_width = st.slider("Largeur", 10, 30, 20)
        grid_height = st.slider("Hauteur", 10, 30, 20)
        
        # Populations
        st.subheader("🦁 Populations d'Animaux")
        mice_count = st.number_input("🐭 Souris", 0, 100, 20)
        cows_count = st.number_input("🐄 Vaches", 0, 50, 8)
        lions_count = st.number_input("🦁 Lions", 0, 20, 4)
        dragons_count = st.number_input("🐲 Dragons", 0, 10, 2)
        
        # Ressources
        st.subheader("🌿 Ressources")
        herbs_count = st.number_input("🌿 Herbes", 0, 100, 25)
        waters_count = st.number_input("💧 Points d'eau", 0, 50, 12)
        
        # Presets
        st.subheader("🎨 Presets")
        preset = st.selectbox("Configurations prédéfinies", [
            "Configuration Personnalisée",
            "Équilibré",
            "Prédateurs Dominants", 
            "Herbivores Paisibles",
            "Survie Extrême"
        ])
        
        if preset != "Configuration Personnalisée":
            if preset == "Équilibré":
                mice_count, cows_count, lions_count, dragons_count = 20, 8, 4, 2
                herbs_count, waters_count = 25, 12
            elif preset == "Prédateurs Dominants":
                mice_count, cows_count, lions_count, dragons_count = 30, 15, 10, 5
                herbs_count, waters_count = 40, 20
            elif preset == "Herbivores Paisibles":
                mice_count, cows_count, lions_count, dragons_count = 40, 20, 2, 1
                herbs_count, waters_count = 50, 25
            elif preset == "Survie Extrême":
                mice_count, cows_count, lions_count, dragons_count = 15, 6, 6, 3
                herbs_count, waters_count = 10, 5
        
        # Boutons de contrôle
        st.subheader("🎮 Contrôles")
        
        if st.button("🚀 Créer Écosystème"):
            ecosystem = EcosystemSimulator(grid_width, grid_height)
            ecosystem.add_animal('Mouse', mice_count)
            ecosystem.add_animal('Cow', cows_count)
            ecosystem.add_animal('Lion', lions_count)
            ecosystem.add_animal('Dragon', dragons_count)
            ecosystem.add_resources(herbs_count, waters_count)
            st.session_state.ecosystem = ecosystem
            st.session_state.running = False
            st.session_state.last_update = time.time()
            st.success("🎉 Écosystème créé!")
            st.rerun()
            
        if st.session_state.ecosystem:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶️ Play" if not st.session_state.running else "⏸️ Pause"):
                    st.session_state.running = not st.session_state.running
                    st.session_state.last_update = time.time()
                    st.rerun()
                    
            with col2:
                if st.button("⏭️ Tour Suivant"):
                    st.session_state.ecosystem.simulate_turn()
                    st.rerun()
            
            # Vitesse de simulation
            st.subheader("⚡ Vitesse")
            speed = st.select_slider(
                "Vitesse de simulation",
                options=[0.5, 1.0, 2.0, 3.0],
                value=st.session_state.auto_speed,
                format_func=lambda x: f"{x}x"
            )
            if speed != st.session_state.auto_speed:
                st.session_state.auto_speed = speed
            
            # Status
            if st.session_state.running:
                st.markdown('<p class="status-running">🟢 Simulation en cours...</p>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-paused">🟡 Simulation en pause</p>', 
                           unsafe_allow_html=True)
        
        # Informations
        st.subheader("📖 À Propos")
        st.markdown("""
        **Simulateur d'Écosystème**
        
        🎯 **Objectif**: Observer l'évolution d'un écosystème virtuel
        
        🔧 **Fonctionnalités**:
        - Prédation réaliste
        - Reproduction naturelle  
        - Gestion des ressources
        - Vieillissement
        - Statistiques temps réel
        
        👨‍💻 **Développé par**: Nicodème KONE
        🔗 **Mail**: nicoetude@gmail.com
        """)
    
    # Gestion de l'auto-refresh
    if st.session_state.ecosystem and st.session_state.running:
        current_time = time.time()
        time_since_last_update = current_time - st.session_state.last_update
        expected_delay = 1.0 / st.session_state.auto_speed
        
        if time_since_last_update >= expected_delay:
            st.session_state.ecosystem.simulate_turn()
            st.session_state.last_update = current_time
            time.sleep(0.1)  # Petite pause pour éviter la surcharge
            st.rerun()
    
    # Zone principale
    if st.session_state.ecosystem:
        ecosystem = st.session_state.ecosystem
        
        # Statistiques en temps réel
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_animals = len(ecosystem.animals)
            st.metric("🦁 Total Animaux", total_animals)
            
        with col2:
            total_resources = len(ecosystem.resources)
            st.metric("🌿 Total Ressources", total_resources)
            
        with col3:
            st.metric("🔄 Tour Actuel", ecosystem.turn)
            
        with col4:
            density = ((total_animals + total_resources) * 100) // (ecosystem.width * ecosystem.height)
            st.metric("📊 Densité", f"{density}%")
        
        # Grille de l'écosystème
        st.subheader("🌍 Planète en Temps Réel")
        
        grid_data = ecosystem.get_grid_data()
        
        # Créer la visualisation avec Plotly
        fig = go.Figure()
        
        # Ajouter les cellules
        for y in range(ecosystem.height):
            for x in range(ecosystem.width):
                cell_content = grid_data[y][x]
                if cell_content and cell_content != 0:
                    # Déterminer la couleur selon le type
                    color = '#8FBC8F'  # Couleur par défaut
                    if cell_content in ['🌿']:
                        color = '#32CD32'
                    elif cell_content in ['💧']:
                        color = '#1E90FF'
                    elif cell_content in ['🐭']:
                        color = '#8D6E63'
                    elif cell_content in ['🐄']:
                        color = '#6D4C41'
                    elif cell_content in ['🦁']:
                        color = '#FF8F00'
                    elif cell_content in ['🐲']:
                        color = '#D32F2F'
                    
                    fig.add_trace(go.Scatter(
                        x=[x], y=[ecosystem.height - 1 - y],  # Inverser Y pour affichage correct
                        mode='markers+text',
                        text=[cell_content],
                        textfont=dict(size=20),
                        marker=dict(size=30, color=color, opacity=0.7),
                        showlegend=False,
                        hovertemplate=f'{cell_content}<br>Position: ({x}, {y})<extra></extra>'
                    ))
        
        # Configuration du graphique
        fig.update_layout(
            title="🌍 Écosystème Vivant",
            xaxis=dict(range=[-0.5, ecosystem.width - 0.5], showgrid=True, gridcolor='lightgray'),
            yaxis=dict(range=[-0.5, ecosystem.height - 0.5], showgrid=True, gridcolor='lightgray'),
            width=800,
            height=600,
            plot_bgcolor='rgba(240,248,255,0.8)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Détails des populations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🦁 Populations Actuelles")
            for species in ecosystem.animal_configs:
                count = len([a for a in ecosystem.animals if a.species == species])
                emoji = ecosystem.animal_configs[species]['emoji']
                st.write(f"{emoji} **{species}**: {count} individus")
                
                if count > 0:
                    # Statistiques de santé
                    animals_of_species = [a for a in ecosystem.animals if a.species == species]
                    avg_life = sum(a.life for a in animals_of_species) / len(animals_of_species)
                    avg_age = sum(a.age for a in animals_of_species) / len(animals_of_species)
                    st.write(f"   ❤️ Vie moyenne: {avg_life:.1f}")
                    st.write(f"   👴 Âge moyen: {avg_age:.1f} ans")
        
        with col2:
            st.subheader("🌿 Ressources Disponibles")
            herbs = len([r for r in ecosystem.resources if r['type'] == 'Herb'])
            water = len([r for r in ecosystem.resources if r['type'] == 'Water'])
            
            st.write(f"🌿 **Herbes**: {herbs} unités")
            st.write(f"💧 **Eau**: {water} unités")
            st.write(f"📊 **Total**: {herbs + water} ressources")
        
        # Graphiques d'évolution
        if len(ecosystem.stats_history) > 1:
            st.subheader("📈 Évolution des Populations")
            
            # Préparer les données pour le graphique
            df_stats = pd.DataFrame(ecosystem.stats_history)
            
            # Graphique des populations
            fig_pop = px.line(df_stats, x='turn', 
                             y=['Mouse', 'Cow', 'Lion', 'Dragon'],
                             title="Évolution des Populations Animales",
                             labels={'value': 'Nombre d\'individus', 'turn': 'Tour'})
            
            fig_pop.update_layout(height=400)
            st.plotly_chart(fig_pop, use_container_width=True)
            
            # Graphique des ressources
            fig_res = px.line(df_stats, x='turn',
                             y=['Herbs', 'Water'],
                             title="Évolution des Ressources",
                             labels={'value': 'Quantité', 'turn': 'Tour'})
            
            fig_res.update_layout(height=300)
            st.plotly_chart(fig_res, use_container_width=True)
            
        # Logs et événements
        if ecosystem.turn > 0:
            st.subheader("📜 Journal de l'Écosystème")
            
            recent_stats = ecosystem.stats_history[-1] if ecosystem.stats_history else {}
            
            st.write(f"**Tour {ecosystem.turn}**:")
            st.write(f"- Population totale: {total_animals} animaux")
            st.write(f"- Ressources disponibles: {total_resources}")
            
            if ecosystem.turn % 10 == 0:
                st.write("- 🎂 Vieillissement des animaux")
            if ecosystem.turn % 15 == 0:
                st.write("- 👶 Saison de reproduction")
            if ecosystem.turn % 5 == 0:
                st.write("- 🌱 Respawn des ressources")
                
        # Force le refresh si en cours d'exécution
        if st.session_state.running:
            time.sleep(0.01)
            st.rerun()
                
    else:
        # Écran d'accueil
        st.subheader("🌟 Bienvenue dans le Simulateur d'Écosystème!")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ### 🎯 Créez et observez un écosystème vivant !
            
            **Fonctionnalités**:
            - 🦁 **4 espèces d'animaux** avec comportements uniques
            - 🌿 **Système de ressources** dynamique
            - ⚔️ **Prédation réaliste** entre espèces
            - 👶 **Reproduction naturelle** des populations
            - 📊 **Statistiques temps réel** et graphiques d'évolution
            - 🎨 **Presets configurables** pour différents scénarios
            
            **Instructions**:
            1. **Configurez votre écosystème dans la sidebar (>>) en haut à droite ⬅️**
            2. Cliquez sur "🚀 Créer Écosystème"
            3. Utilisez "▶️ Play" pour lancer la simulation
            4. Observez l'évolution en temps réel !
            5. Les statistiques sont tout en bas.
            
            ---
            
            *Développé avec ❤️ en Python + Streamlit*
            """)
        
        # Exemple de mini-écosystème statique
        st.subheader("👀 Aperçu de l'Interface")
        
        # Créer un exemple statique
        example_data = {
            'Species': ['🐭 Souris', '🐄 Vaches', '🦁 Lions', '🐲 Dragons', '🌿 Herbes', '💧 Eau'],
            'Count': [18, 6, 3, 2, 22, 10],
            'Status': ['Stable', 'En croissance', 'Chasse active', 'Dominant', 'Abondante', 'Suffisante']
        }
        
        df_example = pd.DataFrame(example_data)
        st.dataframe(df_example, use_container_width=True)


if __name__ == "__main__":
    main()
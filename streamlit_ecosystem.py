# ecosysteme_simulator.py - Version améliorée avec rapport d'analyse
# Simulation par batch (pas de mise en veille) + rapport écologique final

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
from io import BytesIO

# ─── Configuration page ───
st.set_page_config(
    page_title="🌍 Simulateur d'Écosystème",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

    .main-header {
        font-family: 'Nunito', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #2E7D32, #66BB6A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .report-section {
        background: linear-gradient(135deg, #f8fffe, #f0f7f4);
        border-left: 4px solid #2E7D32;
        padding: 1.2rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    .report-warning {
        background: linear-gradient(135deg, #fffbf0, #fff3e0);
        border-left: 4px solid #FF9800;
        padding: 1.2rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    .report-danger {
        background: linear-gradient(135deg, #fff5f5, #ffebee);
        border-left: 4px solid #D32F2F;
        padding: 1.2rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
    }
    .sim-complete {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MODÈLE : Classes de simulation
# ══════════════════════════════════════════════════════════════════

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
        self.cause_of_death = None  # Tracking des causes de mort

    def move(self, grid_width, grid_height):
        new_x = max(0, min(grid_width - 1, self.x + self.direction_x))
        new_y = max(0, min(grid_height - 1, self.y + self.direction_y))
        if random.random() < 0.3:
            self.direction_x = random.choice([-1, 0, 1])
            self.direction_y = random.choice([-1, 0, 1])
        self.x, self.y = new_x, new_y

    def lose_energy(self, amount=1):
        self.life = max(0, self.life - amount)
        if self.life == 0 and self.cause_of_death is None:
            self.cause_of_death = "famine"

    def gain_energy(self, amount=5):
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

        # Tracking avancé pour le rapport
        self.events_log = []         # Journal d'événements
        self.death_tracker = {}      # Causes de mort par espèce
        self.birth_tracker = {}      # Naissances par espèce
        self.predation_tracker = {}  # Prédations : predator -> prey -> count
        self.extinction_turns = {}   # Tour d'extinction par espèce
        self.initial_populations = {}

        self.animal_configs = {
            'Mouse':  {'emoji': '🐭', 'color': '#8D6E63', 'max_life': 15, 'energy_cost': 1,
                       'role': 'Proie', 'trophic_level': 1, 'label_fr': 'Souris'},
            'Cow':    {'emoji': '🐄', 'color': '#6D4C41', 'max_life': 40, 'energy_cost': 1,
                       'role': 'Herbivore', 'trophic_level': 1, 'label_fr': 'Vache'},
            'Lion':   {'emoji': '🦁', 'color': '#FF8F00', 'max_life': 50, 'energy_cost': 2,
                       'role': 'Prédateur', 'trophic_level': 2, 'label_fr': 'Lion'},
            'Dragon': {'emoji': '🐲', 'color': '#D32F2F', 'max_life': 100, 'energy_cost': 1,
                       'role': 'Super-prédateur', 'trophic_level': 3, 'label_fr': 'Dragon'},
        }

    # ── Initialisation ──
    def add_animal(self, species, count):
        config = self.animal_configs[species]
        for _ in range(count):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            life = random.randint(config['max_life'] // 2, config['max_life'])
            self.animals.append(StreamlitAnimal(species, x, y, life, config['max_life']))
        self.initial_populations[species] = count
        self.death_tracker[species] = {'famine': 0, 'predation': 0, 'vieillesse': 0}
        self.birth_tracker[species] = 0
        self.predation_tracker[species] = {}

    def add_resources(self, herb_count, water_count):
        self.resources = []
        for _ in range(herb_count):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            self.resources.append({'type': 'Herb', 'x': x, 'y': y, 'emoji': '🌿', 'value': 5})
        for _ in range(water_count):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            self.resources.append({'type': 'Water', 'x': x, 'y': y, 'emoji': '💧', 'value': 10})

    # ── Boucle de simulation ──
    def simulate_turn(self):
        self.turn += 1

        for animal in self.animals:
            if animal.is_alive():
                animal.move(self.width, self.height)
                animal.lose_energy(self.animal_configs[animal.species]['energy_cost'])

                for resource in self.resources[:]:
                    if animal.x == resource['x'] and animal.y == resource['y']:
                        animal.gain_energy(resource['value'])
                        self.resources.remove(resource)
                        break

        if self.turn % 10 == 0:
            for animal in self.animals:
                if animal.is_alive():
                    animal.age_one_year()
                    if animal.age > 20 and random.random() < 0.1:
                        animal.life = 0
                        animal.cause_of_death = "vieillesse"
                        self.death_tracker[animal.species]['vieillesse'] += 1

        if self.turn % 15 == 0:
            self._reproduction()

        self._predation()

        # Comptabiliser les morts par famine
        for animal in self.animals:
            if not animal.is_alive() and animal.cause_of_death == "famine":
                self.death_tracker[animal.species]['famine'] += 1
                animal.cause_of_death = "comptée"  # éviter double-comptage

        # Détecter extinctions
        for species in self.animal_configs:
            alive = [a for a in self.animals if a.species == species and a.is_alive()]
            if len(alive) == 0 and species not in self.extinction_turns:
                if self.initial_populations.get(species, 0) > 0:
                    self.extinction_turns[species] = self.turn
                    cfg = self.animal_configs[species]
                    self.events_log.append(
                        f"Tour {self.turn}: {cfg['emoji']} **{cfg['label_fr']}** — EXTINCTION"
                    )

        self.animals = [a for a in self.animals if a.is_alive()]

        if self.turn % 5 == 0:
            self._respawn_resources()

        self._save_stats()

    def _predation(self):
        predation_rules = {
            'Dragon': ['Lion', 'Cow', 'Mouse'],
            'Lion': ['Cow', 'Mouse']
        }
        for predator in self.animals:
            if not predator.is_alive() or predator.species not in predation_rules:
                continue
            for prey in self.animals:
                if (prey.species in predation_rules[predator.species]
                        and prey.is_alive()
                        and abs(predator.x - prey.x) <= 1
                        and abs(predator.y - prey.y) <= 1):
                    if random.random() < 0.3:
                        prey.life = 0
                        prey.cause_of_death = "predation"
                        self.death_tracker[prey.species]['predation'] += 1
                        predator.gain_energy(20)
                        # Tracker prédation
                        if prey.species not in self.predation_tracker[predator.species]:
                            self.predation_tracker[predator.species][prey.species] = 0
                        self.predation_tracker[predator.species][prey.species] += 1
                        break

    def _reproduction(self):
        species_groups = {}
        for animal in self.animals:
            if animal.is_alive() and animal.life > animal.max_life * 0.7:
                species_groups.setdefault(animal.species, []).append(animal)

        for species, group in species_groups.items():
            if len(group) >= 2:
                parent1, parent2 = random.sample(group, 2)
                if random.random() < 0.2:
                    config = self.animal_configs[species]
                    baby = StreamlitAnimal(
                        species, parent1.x, parent1.y,
                        config['max_life'] // 2, config['max_life']
                    )
                    self.animals.append(baby)
                    parent1.lose_energy(15)
                    parent2.lose_energy(15)
                    self.birth_tracker[species] = self.birth_tracker.get(species, 0) + 1

    def _respawn_resources(self):
        current_herbs = len([r for r in self.resources if r['type'] == 'Herb'])
        current_water = len([r for r in self.resources if r['type'] == 'Water'])
        for _ in range(max(0, 15 - current_herbs)):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            self.resources.append({'type': 'Herb', 'x': x, 'y': y, 'emoji': '🌿', 'value': 5})
        for _ in range(max(0, 8 - current_water)):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            self.resources.append({'type': 'Water', 'x': x, 'y': y, 'emoji': '💧', 'value': 10})

    def _save_stats(self):
        stats = {'turn': self.turn}
        for species in self.animal_configs:
            count = len([a for a in self.animals if a.species == species and a.is_alive()])
            stats[species] = count
        stats['Herbs'] = len([r for r in self.resources if r['type'] == 'Herb'])
        stats['Water'] = len([r for r in self.resources if r['type'] == 'Water'])
        stats['Total_Animals'] = len(self.animals)
        self.stats_history.append(stats)

    def get_grid_data(self):
        grid = np.zeros((self.height, self.width), dtype=object)
        for resource in self.resources:
            if 0 <= resource['y'] < self.height and 0 <= resource['x'] < self.width:
                grid[resource['y']][resource['x']] = resource['emoji']
        for animal in self.animals:
            if animal.is_alive() and 0 <= animal.y < self.height and 0 <= animal.x < self.width:
                grid[animal.y][animal.x] = self.animal_configs[animal.species]['emoji']
        return grid

    # ── Simulation batch (pas de mise en veille) ──
    def run_batch(self, max_turns, progress_callback=None):
        """Exécute toute la simulation d'un coup, tour par tour."""
        for i in range(max_turns):
            self.simulate_turn()

            # Condition d'arrêt : tous les animaux morts
            if len(self.animals) == 0:
                self.events_log.append(
                    f"Tour {self.turn}: 💀 **Extinction totale** — plus aucun animal vivant."
                )
                break

            if progress_callback:
                progress_callback(i + 1, max_turns)

    # ── Vérification fin ──
    def is_finished(self):
        return len(self.animals) == 0

    # ══════════════════════════════════════════════
    # RAPPORT D'ANALYSE ÉCOLOGIQUE
    # ══════════════════════════════════════════════

    def generate_report(self):
        """Génère un rapport d'analyse complet."""
        if not self.stats_history:
            return None

        df = pd.DataFrame(self.stats_history)
        species_list = list(self.animal_configs.keys())
        report = {}

        # ─ Résumé général ─
        report['duration'] = self.turn
        report['final_populations'] = {s: df[s].iloc[-1] for s in species_list}
        report['initial_populations'] = dict(self.initial_populations)
        report['total_survivors'] = sum(report['final_populations'].values())
        report['extinctions'] = dict(self.extinction_turns)
        report['total_births'] = dict(self.birth_tracker)
        report['total_deaths'] = dict(self.death_tracker)
        report['predation_matrix'] = dict(self.predation_tracker)

        # ─ Indice de biodiversité (Shannon) ─
        final_pops = [v for v in report['final_populations'].values() if v > 0]
        if final_pops:
            total = sum(final_pops)
            proportions = [p / total for p in final_pops]
            report['shannon_index'] = -sum(p * np.log(p) for p in proportions if p > 0)
            report['species_richness'] = len(final_pops)
        else:
            report['shannon_index'] = 0.0
            report['species_richness'] = 0

        # Shannon initial pour comparaison
        init_pops = [v for v in self.initial_populations.values() if v > 0]
        if init_pops:
            total_init = sum(init_pops)
            props_init = [p / total_init for p in init_pops]
            report['shannon_initial'] = -sum(p * np.log(p) for p in props_init if p > 0)
        else:
            report['shannon_initial'] = 0.0

        # ─ Stabilité des populations (coefficient de variation sur 2e moitié) ─
        half = len(df) // 2
        report['stability'] = {}
        for species in species_list:
            series = df[species].iloc[half:]
            mean_val = series.mean()
            if mean_val > 0:
                cv = series.std() / mean_val
                report['stability'][species] = round(cv, 3)
            else:
                report['stability'][species] = None  # éteint

        # ─ Tendances (régression linéaire simple sur chaque espèce) ─
        report['trends'] = {}
        for species in species_list:
            y = df[species].values
            if len(y) > 5:
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0]
                report['trends'][species] = round(slope, 4)
            else:
                report['trends'][species] = 0

        # ─ Pic de population et tours ─
        report['peaks'] = {}
        for species in species_list:
            max_val = df[species].max()
            max_turn = df.loc[df[species] == max_val, 'turn'].iloc[0]
            report['peaks'][species] = {'value': int(max_val), 'turn': int(max_turn)}

        # ─ Ratio prédateurs/proies dans le temps ─
        predator_species = ['Lion', 'Dragon']
        prey_species = ['Mouse', 'Cow']
        df['predators'] = df[predator_species].sum(axis=1)
        df['prey'] = df[prey_species].sum(axis=1)
        df['pred_prey_ratio'] = df.apply(
            lambda r: r['predators'] / r['prey'] if r['prey'] > 0 else np.nan, axis=1
        )
        report['avg_pred_prey_ratio'] = round(df['pred_prey_ratio'].mean(), 3) if df['pred_prey_ratio'].notna().any() else None
        report['df'] = df  # on le garde pour les graphiques

        # ─ Score de santé global ─
        score = 0
        # Biodiversité maintenue ?
        if report['species_richness'] >= 3:
            score += 30
        elif report['species_richness'] >= 2:
            score += 15
        # Pas d'extinction ?
        score += max(0, 25 - len(report['extinctions']) * 10)
        # Stabilité ?
        stabilities = [v for v in report['stability'].values() if v is not None]
        if stabilities:
            avg_stability = np.mean(stabilities)
            if avg_stability < 0.3:
                score += 25
            elif avg_stability < 0.6:
                score += 15
            else:
                score += 5
        # Équilibre prédateurs/proies ?
        if report['avg_pred_prey_ratio'] is not None:
            if 0.1 <= report['avg_pred_prey_ratio'] <= 0.5:
                score += 20
            elif report['avg_pred_prey_ratio'] <= 1.0:
                score += 10

        report['health_score'] = min(100, score)

        return report


# ══════════════════════════════════════════════════════════════════
# AFFICHAGE : Fonctions de rendu
# ══════════════════════════════════════════════════════════════════

def render_grid(ecosystem):
    """Affiche la grille de l'écosystème."""
    grid_data = ecosystem.get_grid_data()
    fig = go.Figure()

    for y in range(ecosystem.height):
        for x in range(ecosystem.width):
            cell = grid_data[y][x]
            if cell and cell != 0:
                color_map = {
                    '🌿': '#32CD32', '💧': '#1E90FF', '🐭': '#8D6E63',
                    '🐄': '#6D4C41', '🦁': '#FF8F00', '🐲': '#D32F2F'
                }
                color = color_map.get(cell, '#8FBC8F')
                fig.add_trace(go.Scatter(
                    x=[x], y=[ecosystem.height - 1 - y],
                    mode='markers+text', text=[cell],
                    textfont=dict(size=18),
                    marker=dict(size=26, color=color, opacity=0.7),
                    showlegend=False,
                    hovertemplate=f'{cell}<br>({x}, {y})<extra></extra>'
                ))

    fig.update_layout(
        xaxis=dict(range=[-0.5, ecosystem.width - 0.5], showgrid=True,
                   gridcolor='rgba(200,200,200,0.3)', title=None),
        yaxis=dict(range=[-0.5, ecosystem.height - 0.5], showgrid=True,
                   gridcolor='rgba(200,200,200,0.3)', title=None),
        height=550, margin=dict(t=20, b=20, l=20, r=20),
        plot_bgcolor='rgba(245,250,248,0.9)'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_population_chart(df, configs):
    """Graphique d'évolution des populations."""
    species_cols = [s for s in configs.keys() if s in df.columns]
    colors = {s: configs[s]['color'] for s in species_cols}

    fig = go.Figure()
    for species in species_cols:
        cfg = configs[species]
        fig.add_trace(go.Scatter(
            x=df['turn'], y=df[species],
            name=f"{cfg['emoji']} {cfg['label_fr']}",
            line=dict(color=cfg['color'], width=2.5),
            fill='tonexty' if species == species_cols[0] else None,
            mode='lines'
        ))

    fig.update_layout(
        title="Évolution des populations",
        xaxis_title="Tour", yaxis_title="Individus",
        height=380, template="plotly_white",
        legend=dict(orientation="h", y=-0.15)
    )
    return fig


def render_report(report, configs):
    """Affiche le rapport d'analyse écologique complet."""
    st.markdown("---")
    st.markdown('<h2 style="text-align:center;">📋 Rapport d\'Analyse Écologique</h2>',
                unsafe_allow_html=True)

    # ── Score de santé ──
    score = report['health_score']
    if score >= 70:
        score_color, score_label = "#4CAF50", "Écosystème sain"
        css_class = "report-section"
    elif score >= 40:
        score_color, score_label = "#FF9800", "Écosystème fragile"
        css_class = "report-warning"
    else:
        score_color, score_label = "#D32F2F", "Écosystème en danger"
        css_class = "report-danger"

    st.markdown(f"""
    <div class="{css_class}">
        <h3 style="margin:0;">Score de santé : <span style="color:{score_color}; font-size:2rem;">{score}/100</span></h3>
        <p style="margin:0.3rem 0 0 0; font-size:1.1rem;">{score_label}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Résumé ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱️ Durée", f"{report['duration']} tours")
    col2.metric("🐾 Survivants", report['total_survivors'])
    col3.metric("🌿 Richesse spécifique", f"{report['species_richness']} / {len(configs)}")
    col4.metric("📊 Indice de Shannon", f"{report['shannon_index']:.2f}")

    # ── Bilan par espèce ──
    st.subheader("🔬 Bilan par espèce")
    species_data = []
    for species, cfg in configs.items():
        init = report['initial_populations'].get(species, 0)
        final = report['final_populations'].get(species, 0)
        births = report['total_births'].get(species, 0)
        deaths = report['total_deaths'].get(species, {})
        total_deaths = sum(deaths.values())
        trend = report['trends'].get(species, 0)
        peak = report['peaks'].get(species, {})

        if trend > 0.05:
            trend_icon = "📈 Croissance"
        elif trend < -0.05:
            trend_icon = "📉 Déclin"
        else:
            trend_icon = "➡️ Stable"

        if species in report['extinctions']:
            status = f"💀 Éteint (tour {report['extinctions'][species]})"
        elif final > init:
            status = "🟢 Prospère"
        elif final > 0:
            status = "🟡 Survie"
        else:
            status = "🔴 Disparue"

        species_data.append({
            'Espèce': f"{cfg['emoji']} {cfg['label_fr']}",
            'Rôle': cfg['role'],
            'Pop. initiale': init,
            'Pop. finale': final,
            'Naissances': births,
            'Morts (total)': total_deaths,
            '  - Famine': deaths.get('famine', 0),
            '  - Prédation': deaths.get('predation', 0),
            '  - Vieillesse': deaths.get('vieillesse', 0),
            'Pic': f"{peak.get('value', '-')} (tour {peak.get('turn', '-')})",
            'Tendance': trend_icon,
            'Statut': status,
        })

    df_species = pd.DataFrame(species_data)
    st.dataframe(df_species, use_container_width=True, hide_index=True)

    # ── Graphiques d'analyse ──
    df = report['df']
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Dynamique des populations")
        fig_pop = render_population_chart(df, configs)
        st.plotly_chart(fig_pop, use_container_width=True)

    with col_right:
        st.subheader("⚖️ Ratio prédateurs / proies")
        if 'pred_prey_ratio' in df.columns:
            fig_ratio = go.Figure()
            fig_ratio.add_trace(go.Scatter(
                x=df['turn'], y=df['pred_prey_ratio'],
                mode='lines', name='Ratio',
                line=dict(color='#7B1FA2', width=2)
            ))
            fig_ratio.add_hline(y=0.3, line_dash="dash", line_color="green",
                                annotation_text="Ratio idéal (~0.3)")
            fig_ratio.update_layout(
                xaxis_title="Tour", yaxis_title="Ratio P/p",
                height=380, template="plotly_white"
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

    # ── Chaîne alimentaire ──
    st.subheader("🔗 Interactions trophiques observées")
    pred_data = report['predation_matrix']
    has_predation = any(v for v in pred_data.values() if v)

    if has_predation:
        rows = []
        for predator, preys in pred_data.items():
            for prey, count in preys.items():
                if count > 0:
                    p_cfg = configs[predator]
                    r_cfg = configs[prey]
                    rows.append({
                        'Prédateur': f"{p_cfg['emoji']} {p_cfg['label_fr']}",
                        'Proie': f"{r_cfg['emoji']} {r_cfg['label_fr']}",
                        'Prédations': count
                    })
        if rows:
            df_pred = pd.DataFrame(rows).sort_values('Prédations', ascending=False)
            st.dataframe(df_pred, use_container_width=True, hide_index=True)

            # Sankey diagram
            labels, sources, targets, values = [], [], [], []
            label_map = {}
            for _, row in df_pred.iterrows():
                for name in [row['Prédateur'], row['Proie']]:
                    if name not in label_map:
                        label_map[name] = len(labels)
                        labels.append(name)
                sources.append(label_map[row['Proie']])
                targets.append(label_map[row['Prédateur']])
                values.append(row['Prédations'])

            fig_sankey = go.Figure(go.Sankey(
                node=dict(pad=20, thickness=20, label=labels,
                          color=["#66BB6A", "#8D6E63", "#FF8F00", "#D32F2F"][:len(labels)]),
                link=dict(source=sources, target=targets, value=values,
                          color="rgba(200,200,200,0.4)")
            ))
            fig_sankey.update_layout(title="Flux d'énergie trophique", height=350)
            st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.info("Aucune prédation observée pendant la simulation.")

    # ── Graphique des causes de mort ──
    st.subheader("💀 Causes de mortalité")
    death_rows = []
    for species, causes in report['total_deaths'].items():
        cfg = configs[species]
        for cause, count in causes.items():
            if count > 0:
                death_rows.append({
                    'Espèce': f"{cfg['emoji']} {cfg['label_fr']}",
                    'Cause': cause.capitalize(),
                    'Nombre': count
                })

    if death_rows:
        df_deaths = pd.DataFrame(death_rows)
        fig_death = px.bar(
            df_deaths, x='Espèce', y='Nombre', color='Cause',
            barmode='stack', color_discrete_map={
                'Famine': '#FF9800', 'Predation': '#D32F2F', 'Vieillesse': '#78909C'
            },
            height=350
        )
        fig_death.update_layout(template="plotly_white")
        st.plotly_chart(fig_death, use_container_width=True)

    # ── Ressources ──
    st.subheader("🌿 Évolution des ressources")
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=df['turn'], y=df['Herbs'], name='🌿 Herbes',
        line=dict(color='#4CAF50', width=2), fill='tozeroy'
    ))
    fig_res.add_trace(go.Scatter(
        x=df['turn'], y=df['Water'], name='💧 Eau',
        line=dict(color='#2196F3', width=2), fill='tozeroy'
    ))
    fig_res.update_layout(
        xaxis_title="Tour", yaxis_title="Quantité",
        height=300, template="plotly_white",
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig_res, use_container_width=True)

    # ── Biodiversité ──
    st.subheader("🌱 Analyse de la biodiversité")
    col1, col2 = st.columns(2)

    with col1:
        delta_shannon = report['shannon_index'] - report['shannon_initial']
        if delta_shannon > -0.1:
            st.success(f"L'indice de Shannon est passé de **{report['shannon_initial']:.2f}** à "
                       f"**{report['shannon_index']:.2f}** — la diversité a été globalement maintenue.")
        else:
            st.warning(f"L'indice de Shannon a chuté de **{report['shannon_initial']:.2f}** à "
                       f"**{report['shannon_index']:.2f}** — perte de biodiversité significative.")

    with col2:
        if report['extinctions']:
            for sp, turn in report['extinctions'].items():
                cfg = configs[sp]
                st.error(f"{cfg['emoji']} **{cfg['label_fr']}** s'est éteint au tour {turn}.")
        else:
            st.success("Aucune extinction pendant la simulation !")

    # ── Interprétation / Enseignements ──
    st.subheader("🎓 Enseignements écologiques")
    lessons = _generate_lessons(report, configs)
    for lesson in lessons:
        st.markdown(f"""<div class="report-section">{lesson}</div>""", unsafe_allow_html=True)


def _generate_lessons(report, configs):
    """Génère des enseignements contextuels basés sur les résultats."""
    lessons = []

    # Leçon sur les extinctions
    if report['extinctions']:
        extinct_species = list(report['extinctions'].keys())
        extinct_roles = [configs[s]['role'] for s in extinct_species]
        if 'Proie' in extinct_roles or 'Herbivore' in extinct_roles:
            lessons.append(
                "🔴 <b>Effondrement des proies</b> — L'extinction des herbivores/proies est souvent causée par "
                "une pression de prédation excessive. Dans un écosystème réel, cela entraînerait ensuite "
                "le déclin des prédateurs (effet en cascade). C'est le principe des <i>cascades trophiques</i>."
            )
        if 'Prédateur' in extinct_roles or 'Super-prédateur' in extinct_roles:
            lessons.append(
                "🟠 <b>Disparition des prédateurs</b> — Sans prédateurs, les populations de proies peuvent exploser "
                "puis s'effondrer par surexploitation des ressources. C'est le concept de <i>régulation top-down</i> "
                "observé dans les écosystèmes naturels (ex: réintroduction des loups à Yellowstone)."
            )

    # Leçon sur l'équilibre prédateur/proie
    ratio = report.get('avg_pred_prey_ratio')
    if ratio is not None:
        if ratio > 0.8:
            lessons.append(
                "⚠️ <b>Trop de prédateurs</b> — Le ratio prédateurs/proies est élevé ({:.2f}). "
                "Dans la nature, un ratio au-delà de 0.5 est rarement soutenable. Les prédateurs exercent "
                "une pression qui finit par détruire leur propre base alimentaire (<i>surexploitation</i>).".format(ratio)
            )
        elif ratio < 0.1 and ratio > 0:
            lessons.append(
                "🟢 <b>Dominance des proies</b> — Peu de prédateurs par rapport aux proies (ratio {:.2f}). "
                "Cela peut mener à une croissance exponentielle des herbivores, puis à un effondrement "
                "des ressources végétales (<i>surpâturage</i>).".format(ratio)
            )

    # Leçon sur la famine
    total_famine = sum(d.get('famine', 0) for d in report['total_deaths'].values())
    total_all_deaths = sum(sum(d.values()) for d in report['total_deaths'].values())
    if total_all_deaths > 0 and total_famine / total_all_deaths > 0.5:
        lessons.append(
            "🍂 <b>Famine dominante</b> — Plus de la moitié des décès sont dus à la famine. "
            "Cela indique une <i>capacité de charge</i> (carrying capacity) insuffisante : "
            "le milieu ne peut pas nourrir autant d'individus. Augmenter les ressources "
            "ou réduire les populations initiales stabiliserait le système."
        )

    # Leçon sur la stabilité
    stabilities = [v for v in report['stability'].values() if v is not None]
    if stabilities:
        avg_stab = np.mean(stabilities)
        if avg_stab < 0.2:
            lessons.append(
                "✅ <b>Populations stables</b> — Les populations ont atteint un état d'équilibre relatif, "
                "ce qui est le signe d'un écosystème bien régulé. En écologie, c'est l'état de "
                "<i>climax</i> ou d'équilibre dynamique."
            )
        elif avg_stab > 0.7:
            lessons.append(
                "🔄 <b>Forte instabilité</b> — Les populations oscillent fortement, typique des "
                "cycles prédateur-proie décrits par le modèle de <i>Lotka-Volterra</i>. "
                "Ces oscillations peuvent mener à des extinctions si l'amplitude est trop grande."
            )

    # Leçon biodiversité
    if report['shannon_index'] > 1.0 and not report['extinctions']:
        lessons.append(
            "🌈 <b>Biodiversité préservée</b> — Un indice de Shannon élevé sans extinction "
            "montre un écosystème résilient. La diversité des espèces renforce la stabilité : "
            "c'est l'<i>hypothèse de l'assurance écologique</i> (insurance hypothesis)."
        )

    if not lessons:
        lessons.append(
            "📝 <b>Simulation courte</b> — La simulation n'a pas duré assez longtemps pour "
            "observer des dynamiques écologiques marquées. Essayez avec plus de tours (200+) "
            "pour voir les cycles prédateur-proie et les effets de compétition se développer."
        )

    return lessons


def generate_text_report(report, configs):
    """Génère un rapport texte téléchargeable."""
    lines = []
    lines.append("=" * 60)
    lines.append("   RAPPORT D'ANALYSE ÉCOLOGIQUE")
    lines.append(f"   Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Score de santé : {report['health_score']}/100")
    lines.append(f"Durée : {report['duration']} tours")
    lines.append(f"Survivants : {report['total_survivors']}")
    lines.append(f"Indice de Shannon : {report['shannon_index']:.3f} (initial: {report['shannon_initial']:.3f})")
    lines.append(f"Richesse spécifique : {report['species_richness']} / {len(configs)}")
    lines.append("")

    lines.append("-" * 40)
    lines.append("BILAN PAR ESPÈCE")
    lines.append("-" * 40)
    for species, cfg in configs.items():
        init = report['initial_populations'].get(species, 0)
        final = report['final_populations'].get(species, 0)
        births = report['total_births'].get(species, 0)
        deaths = report['total_deaths'].get(species, {})
        total_deaths = sum(deaths.values())
        peak = report['peaks'].get(species, {})
        lines.append(f"\n{cfg['label_fr']} ({cfg['role']})")
        lines.append(f"  Initial: {init}  |  Final: {final}  |  Naissances: {births}  |  Morts: {total_deaths}")
        lines.append(f"  Morts par famine: {deaths.get('famine',0)} | prédation: {deaths.get('predation',0)} | vieillesse: {deaths.get('vieillesse',0)}")
        lines.append(f"  Pic: {peak.get('value','-')} au tour {peak.get('turn','-')}")
        if species in report['extinctions']:
            lines.append(f"  *** ÉTEINT au tour {report['extinctions'][species]} ***")

    if report['extinctions']:
        ext_parts = []
        for s, t in report['extinctions'].items():
            label = configs[s]['label_fr']
            ext_parts.append(f"{label} (tour {t})")
        lines.append(f"\n\nEXTINCTIONS : {', '.join(ext_parts)}")

    ratio = report.get('avg_pred_prey_ratio')
    if ratio is not None:
        lines.append(f"\nRatio moyen prédateurs/proies : {ratio:.3f}")

    lines.append("\n" + "=" * 60)
    lines.append("   Fin du rapport")
    lines.append("=" * 60)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# APPLICATION STREAMLIT
# ══════════════════════════════════════════════════════════════════

def _render_live_view(ecosystem):
    """Affiche la vue en temps réel pendant l'animation (grille + stats + graphiques)."""

    # Métriques en haut
    col1, col2, col3, col4 = st.columns(4)
    total_animals = len(ecosystem.animals)
    total_resources = len(ecosystem.resources)
    col1.metric("🦁 Animaux", total_animals)
    col2.metric("🌿 Ressources", total_resources)
    col3.metric("🔄 Tour", f"{ecosystem.turn} / {st.session_state.max_turns}")
    density = ((total_animals + total_resources) * 100) // max(1, ecosystem.width * ecosystem.height)
    col4.metric("📊 Densité", f"{density}%")

    # Barre de progression
    progress_pct = ecosystem.turn / max(1, st.session_state.max_turns)
    st.progress(progress_pct, text=f"Tour {ecosystem.turn} sur {st.session_state.max_turns}")

    # Grille
    st.subheader("🌍 Planète en temps réel")
    render_grid(ecosystem)

    # Détails populations + ressources
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("🦁 Populations")
        for species, cfg in ecosystem.animal_configs.items():
            count = len([a for a in ecosystem.animals if a.species == species])
            if count > 0:
                animals_sp = [a for a in ecosystem.animals if a.species == species]
                avg_life = sum(a.life for a in animals_sp) / len(animals_sp)
                avg_age = sum(a.age for a in animals_sp) / len(animals_sp)
                st.write(f"{cfg['emoji']} **{cfg['label_fr']}** : {count}  —  ❤️ {avg_life:.0f}  👴 {avg_age:.0f}")
            else:
                st.write(f"{cfg['emoji']} **{cfg['label_fr']}** : 💀 éteint")

    with col_right:
        st.subheader("🌿 Ressources")
        herbs = len([r for r in ecosystem.resources if r['type'] == 'Herb'])
        water = len([r for r in ecosystem.resources if r['type'] == 'Water'])
        st.write(f"🌿 **Herbes** : {herbs}")
        st.write(f"💧 **Eau** : {water}")

    # Graphique d'évolution en direct
    if len(ecosystem.stats_history) > 1:
        st.subheader("📈 Évolution des populations")
        df_stats = pd.DataFrame(ecosystem.stats_history)
        fig = render_population_chart(df_stats, ecosystem.animal_configs)
        st.plotly_chart(fig, use_container_width=True)

    # Journal
    if ecosystem.events_log:
        with st.expander("📜 Événements récents", expanded=False):
            for event in ecosystem.events_log[-10:]:
                st.markdown(f"- {event}")


def _check_simulation_end(ecosystem, max_turns):
    """Vérifie si la simulation doit se terminer. Retourne True si finie."""
    if ecosystem.is_finished():
        st.session_state.running = False
        st.session_state.simulation_done = True
        st.session_state.report = ecosystem.generate_report()
        return True
    if ecosystem.turn >= max_turns:
        st.session_state.running = False
        st.session_state.simulation_done = True
        st.session_state.report = ecosystem.generate_report()
        return True
    return False


def main():
    st.markdown('<h1 class="main-header">🌍 Simulateur d\'Écosystème</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Observez, analysez et comprenez les dynamiques écologiques</p>',
                unsafe_allow_html=True)

    # ── Session state ──
    if 'ecosystem' not in st.session_state:
        st.session_state.ecosystem = None
        st.session_state.simulation_done = False
        st.session_state.report = None
        st.session_state.running = False        # pour le mode animé
        st.session_state.sim_mode = "animated"   # animated | batch | step
        st.session_state.max_turns = 200
        st.session_state.auto_speed = 1.0
        st.session_state.last_update = time.time()

    # ══════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🌍 Planète")
        grid_width = st.slider("Largeur", 10, 30, 20)
        grid_height = st.slider("Hauteur", 10, 30, 20)

        st.subheader("🦁 Populations")
        mice_count = st.number_input("🐭 Souris", 0, 100, 20)
        cows_count = st.number_input("🐄 Vaches", 0, 50, 8)
        lions_count = st.number_input("🦁 Lions", 0, 20, 4)
        dragons_count = st.number_input("🐲 Dragons", 0, 10, 2)

        st.subheader("🌿 Ressources")
        herbs_count = st.number_input("🌿 Herbes", 0, 100, 25)
        waters_count = st.number_input("💧 Points d'eau", 0, 50, 12)

        st.subheader("⏱️ Durée de simulation")
        max_turns = st.slider("Nombre de tours max", 50, 500, 200, step=25,
                              help="La simulation s'arrête automatiquement après ce nombre "
                                   "de tours (ou si tous les animaux meurent).")
        st.session_state.max_turns = max_turns

        st.subheader("🎨 Presets")
        preset = st.selectbox("Scénarios prédéfinis", [
            "Personnalisé", "Équilibré", "Prédateurs Dominants",
            "Herbivores Paisibles", "Survie Extrême"
        ])
        presets_map = {
            "Équilibré": (20, 8, 4, 2, 25, 12),
            "Prédateurs Dominants": (30, 15, 10, 5, 40, 20),
            "Herbivores Paisibles": (40, 20, 2, 1, 50, 25),
            "Survie Extrême": (15, 6, 6, 3, 10, 5),
        }
        if preset in presets_map:
            mice_count, cows_count, lions_count, dragons_count, herbs_count, waters_count = presets_map[preset]

        st.markdown("---")

        # ── Choix du mode ──
        st.subheader("🎮 Mode de simulation")
        mode_choice = st.radio(
            "Comment voulez-vous observer ?",
            ["🎬 Animée (temps réel)", "⚡ Rapide (batch)", "🔬 Pas à pas"],
            help="**Animée** : visualisation en direct tour par tour, comme un film. "
                 "**Rapide** : exécute tout d'un coup, puis affiche le rapport. "
                 "**Pas à pas** : vous contrôlez chaque tour manuellement."
        )
        mode_map = {
            "🎬 Animée (temps réel)": "animated",
            "⚡ Rapide (batch)": "batch",
            "🔬 Pas à pas": "step",
        }
        selected_mode = mode_map[mode_choice]

        # Vitesse (seulement pour le mode animé)
        if selected_mode == "animated":
            st.subheader("⚡ Vitesse d'animation")
            speed = st.select_slider(
                "Tours par seconde",
                options=[0.5, 1.0, 2.0, 3.0, 5.0],
                value=st.session_state.auto_speed,
                format_func=lambda x: f"{x}x"
            )
            if speed != st.session_state.auto_speed:
                st.session_state.auto_speed = speed

        # ── Bouton créer ──
        if st.button("🚀 Créer l'écosystème", type="primary", use_container_width=True):
            eco = EcosystemSimulator(grid_width, grid_height)
            eco.add_animal('Mouse', mice_count)
            eco.add_animal('Cow', cows_count)
            eco.add_animal('Lion', lions_count)
            eco.add_animal('Dragon', dragons_count)
            eco.add_resources(herbs_count, waters_count)

            st.session_state.ecosystem = eco
            st.session_state.simulation_done = False
            st.session_state.report = None
            st.session_state.running = False
            st.session_state.sim_mode = selected_mode
            st.session_state.last_update = time.time()

            if selected_mode == "batch":
                # ─ Mode rapide : on exécute tout d'un coup ─
                progress = st.progress(0, text="Simulation en cours...")

                def update_progress(current, total):
                    progress.progress(current / total,
                                      text=f"Tour {current}/{total} — "
                                           f"{len(eco.animals)} animaux vivants")

                eco.run_batch(max_turns, progress_callback=update_progress)
                progress.empty()

                st.session_state.simulation_done = True
                st.session_state.report = eco.generate_report()

            st.rerun()

        # ── Contrôles en cours de simulation ──
        ecosystem = st.session_state.ecosystem
        if ecosystem and not st.session_state.simulation_done:

            st.markdown("---")

            # Mode animé : Play / Pause
            if st.session_state.sim_mode == "animated":
                col1, col2 = st.columns(2)
                with col1:
                    btn_label = "⏸️ Pause" if st.session_state.running else "▶️ Play"
                    if st.button(btn_label, use_container_width=True):
                        st.session_state.running = not st.session_state.running
                        st.session_state.last_update = time.time()
                        st.rerun()
                with col2:
                    if st.button("⏭️ +1 tour", use_container_width=True):
                        st.session_state.running = False
                        ecosystem.simulate_turn()
                        _check_simulation_end(ecosystem, max_turns)
                        st.rerun()

                if st.button("🏁 Arrêter et voir le rapport", use_container_width=True):
                    st.session_state.running = False
                    st.session_state.simulation_done = True
                    st.session_state.report = ecosystem.generate_report()
                    st.rerun()

                # Status
                if st.session_state.running:
                    st.markdown("🟢 **Simulation en cours...**")
                else:
                    st.markdown("🟡 **En pause**")

            # Mode pas à pas
            elif st.session_state.sim_mode == "step":
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("⏭️ Tour suivant", use_container_width=True):
                        ecosystem.simulate_turn()
                        _check_simulation_end(ecosystem, max_turns)
                        st.rerun()
                with col2:
                    if st.button("⏩ +10 tours", use_container_width=True):
                        for _ in range(10):
                            ecosystem.simulate_turn()
                            if _check_simulation_end(ecosystem, max_turns):
                                break
                        st.rerun()

                if st.button("🏁 Terminer et voir le rapport", use_container_width=True):
                    st.session_state.simulation_done = True
                    st.session_state.report = ecosystem.generate_report()
                    st.rerun()

        # Crédits
        st.markdown("---")
        st.markdown("""
        👨‍💻 **Développé par** : Nicodème KONE
        🔗 **Mail** : nicoetude@gmail.com
        """)

    # ══════════════════════════════════════════════
    # ZONE PRINCIPALE
    # ══════════════════════════════════════════════

    ecosystem = st.session_state.ecosystem

    # ── Écran d'accueil ──
    if ecosystem is None:
        st.subheader("🌟 Bienvenue !")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### Créez et analysez un écosystème virtuel

            Ce simulateur modélise les interactions entre espèces dans un environnement fermé.
            À la fin de la simulation, un **rapport d'analyse écologique** complet est généré
            avec des métriques scientifiques (indice de Shannon, ratio prédateur/proie, cascades trophiques).

            **Trois modes** :
            - **🎬 Animée** : observez l'écosystème évoluer en temps réel avec Play/Pause
            - **⚡ Rapide** : tout est calculé d'un coup, résultat immédiat
            - **🔬 Pas à pas** : avancez tour par tour pour étudier en détail

            Dans tous les cas, la simulation a une **durée limitée** et se termine
            avec un rapport d'analyse complet.

            **Commencez** ➡️ configurez dans le panneau latéral puis cliquez "Créer l'écosystème".

            ---
            *Développé avec Python + Streamlit*
            """)

        st.subheader("📊 Aperçu des métriques du rapport")
        example_data = {
            'Métrique': ['Indice de Shannon', 'Ratio prédateurs/proies', 'Score de santé',
                         'Richesse spécifique', 'Stabilité moyenne'],
            'Description': [
                'Mesure la diversité biologique (0 = 1 seule espèce, >1 = bonne diversité)',
                'Proportion prédateurs vs proies (idéal ≈ 0.3)',
                'Score composite de santé de l\'écosystème (0-100)',
                'Nombre d\'espèces survivantes à la fin',
                'Coefficient de variation des populations (faible = stable)'
            ]
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True, hide_index=True)
        return

    # ── Simulation terminée → Rapport ──
    if st.session_state.simulation_done and st.session_state.report:
        report = st.session_state.report

        st.markdown("""
        <div class="sim-complete">
            <h3 style="margin:0;">✅ Simulation terminée</h3>
            <p style="margin:0.3rem 0 0 0;">Achevée après {} tours — consultez le rapport ci-dessous.</p>
        </div>
        """.format(report['duration']), unsafe_allow_html=True)

        # Grille finale
        with st.expander("🌍 État final de la planète", expanded=False):
            render_grid(ecosystem)

        # Rapport complet
        render_report(report, ecosystem.animal_configs)

        # Téléchargements
        st.markdown("---")
        txt_report = generate_text_report(report, ecosystem.animal_configs)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                "📥 Télécharger le rapport (texte)",
                data=txt_report,
                file_name=f"rapport_ecosysteme_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            df_export = pd.DataFrame(ecosystem.stats_history)
            csv_data = df_export.to_csv(index=False)
            st.download_button(
                "📊 Télécharger les données (CSV)",
                data=csv_data,
                file_name=f"donnees_ecosysteme_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        return

    # ══════════════════════════════════════════════
    # SIMULATION EN COURS (animée ou pas à pas)
    # ══════════════════════════════════════════════

    # Affichage commun : vue en direct
    if st.session_state.sim_mode == "step":
        st.info(f"🔬 **Mode pas à pas** — Tour {ecosystem.turn} / {st.session_state.max_turns}")
    elif st.session_state.sim_mode == "animated":
        if st.session_state.running:
            st.success(f"🎬 **Animation en cours** — Tour {ecosystem.turn} / {st.session_state.max_turns}")
        else:
            st.info(f"🎬 **Animation en pause** — Tour {ecosystem.turn} / {st.session_state.max_turns}  "
                    f"*(Cliquez ▶️ Play dans le panneau latéral)*")

    _render_live_view(ecosystem)

    # ── Auto-refresh pour le mode animé ──
    if st.session_state.sim_mode == "animated" and st.session_state.running:
        current_time = time.time()
        expected_delay = 1.0 / st.session_state.auto_speed

        if current_time - st.session_state.last_update >= expected_delay:
            ecosystem.simulate_turn()
            st.session_state.last_update = current_time

            # Vérifier la fin
            if _check_simulation_end(ecosystem, st.session_state.max_turns):
                st.rerun()

        # Continuer l'animation
        time.sleep(0.05)
        st.rerun()


if __name__ == "__main__":
    main()

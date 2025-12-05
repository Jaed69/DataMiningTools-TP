"""
Módulos de tabs para la aplicación Streamlit.
"""
from .tab_recommender import render_tab_recommender
from .tab_search import render_tab_search
from .tab_classifier import render_tab_classifier
from .tab_evaluation import render_tab_evaluation
from .tab_info import render_tab_info

__all__ = [
    "render_tab_recommender",
    "render_tab_search", 
    "render_tab_classifier",
    "render_tab_evaluation",
    "render_tab_info"
]

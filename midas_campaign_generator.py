#!/usr/bin/env python3
"""
Sistema de Generación Automática de Campañas MIDAS - OMNIA
Desarrollado para automatizar completamente la creación, despliegue y gestión de campañas publicitarias.

Funcionalidades principales:
- Generador de campañas basado en templates dinámicos
- Sistema de targeting automático (geolocalización, demografía, intereses)
- Creación automática de ad groups y keywords
- Generación de ad copy usando AI/LLM
- Cálculo automático de presupuestos por objetivo
- Sistema de programación de campañas
- A/B testing automático de creativos
- Integración con datos de productos desde ANCHOR

Autor: Sistema OMNIA - Módulo MIDAS
Fecha: 2025-11-06
"""

import json
import hashlib
import logging
import uuid
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import random
import asyncio
import requests
from abc import ABC, abstractmethod


# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('midas_campaign_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CampaignObjective(Enum):
    """Objetivos de campaña disponibles"""
    AWARENESS = "awareness"
    TRAFFIC = "traffic"
    ENGAGEMENT = "engagement"
    LEAD_GENERATION = "lead_generation"
    CONVERSIONS = "conversions"
    SALES = "sales"
    APP_INSTALLS = "app_installs"
    VIDEO_VIEWS = "video_views"


class CampaignPlatform(Enum):
    """Plataformas de publicidad soportadas"""
    GOOGLE_ADS = "google_ads"
    META_ADS = "meta_ads"
    LINKEDIN_ADS = "linkedin_ads"
    TIKTOK_ADS = "tiktok_ads"
    TWITTER_ADS = "twitter_ads"


class AdFormat(Enum):
    """Formatos de anuncio disponibles"""
    TEXT_AD = "text_ad"
    IMAGE_AD = "image_ad"
    VIDEO_AD = "video_ad"
    CAROUSEL_AD = "carousel_ad"
    COLLECTION_AD = "collection_ad"


@dataclass
class TargetingCriteria:
    """Criterios de targeting automático"""
    countries: List[str]
    regions: Optional[List[str]] = None
    cities: Optional[List[str]] = None
    age_range: Optional[Tuple[int, int]] = None
    gender: Optional[str] = None  # male, female, all
    interests: Optional[List[str]] = None
    behaviors: Optional[List[str]] = None
    job_titles: Optional[List[str]] = None
    company_size: Optional[List[str]] = None
    industries: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    device_types: Optional[List[str]] = None
    connection_type: Optional[List[str]] = None


@dataclass
class CampaignConfig:
    """Configuración de campaña"""
    cmd_id: str
    intent: str
    market: str
    product_line: str
    budget: float
    duration_days: int
    objective: CampaignObjective
    platform: List[CampaignPlatform]
    ad_formats: List[AdFormat]
    start_date: Optional[datetime.datetime] = None
    end_date: Optional[datetime.datetime] = None
    scheduling: Optional[Dict] = None
    targeting: Optional[TargetingCriteria] = None
    ab_testing: Optional[bool] = True
    anchor_integration: Optional[bool] = True


@dataclass
class Creative:
    """Modelo de creativo publicitario"""
    creative_id: str
    title: str
    description: str
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    call_to_action: str = "Learn More"
    headline: Optional[str] = None
    display_url: Optional[str] = None
    final_url: Optional[str] = None
    ab_variant: Optional[str] = None


@dataclass
class AdGroup:
    """Grupo de anuncios"""
    ad_group_id: str
    name: str
    keywords: List[str]
    creatives: List[Creative]
    bid_strategy: str = "automatic"
    daily_budget: Optional[float] = None
    max_cpc: Optional[float] = None


@dataclass
class Campaign:
    """Campaña publicitaria completa"""
    campaign_id: str
    config: CampaignConfig
    ad_groups: List[AdGroup]
    total_budget: float
    estimated_reach: int
    estimated_clicks: int
    estimated_conversions: int
    created_at: datetime.datetime
    trace_id: str
    status: str = "draft"


class CampaignTemplate:
    """Generador de campañas basado en templates dinámicos"""
    
    def __init__(self, anchor_api_url: str = "https://api.anchor.com", api_key: str = None):
        self.anchor_api_url = anchor_api_url
        self.api_key = api_key
        self.templates = self._load_templates()
        self.anchor_client = AnchorClient(anchor_api_url, api_key)
        
    def _load_templates(self) -> Dict[str, Any]:
        """Carga templates dinámicos de campaña"""
        return {
            "product_launch": {
                "name": "Lanzamiento de Producto",
                "description": "Template para lanzamiento de nuevos productos",
                "ad_formats": [AdFormat.IMAGE_AD, AdFormat.VIDEO_AD, AdFormat.TEXT_AD],
                "default_objective": CampaignObjective.AWARENESS,
                "bid_strategy": "maximize_reach",
                "creative_templates": {
                    "headline": [
                        "Nuevo {product_name}: {benefit}",
                        "Descubre {product_name}",
                        "¡Lanzamiento! {product_name} ya disponible"
                    ],
                    "description": [
                        "Experimenta la innovación de {product_name}. {key_benefit}",
                        "Cambia tu forma de trabajar con {product_name}",
                        "La nueva generación de {product_category} ha llegado"
                    ],
                    "call_to_action": [
                        "Descubre más", "Ver ahora", "Comprar", "Prueba gratis"
                    ]
                },
                "targeting_presets": {
                    "tech_enthusiasts": {
                        "interests": ["Technology", "Innovation", "Gadgets"],
                        "behaviors": ["Early Adopters", "Tech Enthusiasts"],
                        "age_range": (25, 45)
                    },
                    "business_professionals": {
                        "interests": ["Business", "Productivity", "Leadership"],
                        "job_titles": ["Manager", "Director", "CEO", "CTO"],
                        "age_range": (30, 55)
                    }
                }
            },
            "black_friday": {
                "name": "Black Friday",
                "description": "Template para campañas de ofertas especiales",
                "ad_formats": [AdFormat.IMAGE_AD, AdFormat.CAROUSEL_AD, AdFormat.COLLECTION_AD],
                "default_objective": CampaignObjective.CONVERSIONS,
                "bid_strategy": "maximize_conversions",
                "creative_templates": {
                    "headline": [
                        "Black Friday: {discount}% OFF",
                        "¡Oferta limitada! {discount}% de descuento",
                        "Black Friday - {product_category} con {discount}% OFF"
                    ],
                    "description": [
                        "Solo por {duration} - {product_name} con {discount}% de descuento",
                        "La oferta más grande del año. {discount}% OFF en {product_name}",
                        "Black Friday especial: {discount}% de descuento en {category_name}"
                    ],
                    "call_to_action": [
                        "Aprovechar oferta", "Comprar ahora", "Ver descuentos", "No perderselo"
                    ]
                },
                "targeting_presets": {
                    "deal_seekers": {
                        "interests": ["Shopping", "Deals", "Discounts"],
                        "behaviors": ["Deal Hunters", "Online Shoppers"],
                        "age_range": (18, 50)
                    }
                }
            },
            "lead_generation": {
                "name": "Generación de Leads",
                "description": "Template para captura de leads",
                "ad_formats": [AdFormat.TEXT_AD, AdFormat.IMAGE_AD],
                "default_objective": CampaignObjective.LEAD_GENERATION,
                "bid_strategy": "maximize_leads",
                "creative_templates": {
                    "headline": [
                        "Descarga gratuita: {lead_magnet}",
                        "Obtén {benefit} gratis",
                        "Regístrate y recibe {free_item}"
                    ],
                    "description": [
                        "Accede a {lead_magnet} sin costo. ¡Regístrate ahora!",
                        "Descubre cómo {benefit}. Descarga gratuita disponible",
                        "Descarga gratuita de {lead_magnet} - {description}"
                    ],
                    "call_to_action": [
                        "Descargar gratis", "Registrarme", "Obtener acceso", "Descargarlo"
                    ]
                },
                "targeting_presets": {
                    "professionals": {
                        "interests": ["Professional Development", "Career", "Training"],
                        "age_range": (25, 50)
                    }
                }
            },
            "retargeting": {
                "name": "Retargeting",
                "description": "Template para retargeting de usuarios",
                "ad_formats": [AdFormat.IMAGE_AD, AdFormat.CAROUSEL_AD],
                "default_objective": CampaignObjective.CONVERSIONS,
                "bid_strategy": "target_conversion",
                "creative_templates": {
                    "headline": [
                        "¿Aún pensando en {product_name}?",
                        "Te esperamos de vuelta",
                        "Completa tu compra: {product_name}"
                    ],
                    "description": [
                        "No pierdas la oportunidad. {product_name} te está esperando",
                        "Tu carrito te recuerda. Finaliza tu compra hoy",
                        "Especial para ti: {discount}% OFF en {product_name}"
                    ],
                    "call_to_action": [
                        "Completar compra", "Volver al carrito", "Finalizar pedido", "Continuar"
                    ]
                },
                "targeting_presets": {
                    "website_visitors": {
                        "behaviors": ["Website Visitors", "Cart Abandoners"],
                        "age_range": (18, 65)
                    }
                }
            }
        }
    
    async def create_campaign_from_template(
        self, 
        template_name: str, 
        config: CampaignConfig,
        custom_data: Optional[Dict] = None
    ) -> Campaign:
        """
        Crea una campaña usando un template específico
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' no encontrado")
        
        template = self.templates[template_name]
        trace_id = self._generate_trace_id(config)
        
        logger.info(f"Creando campaña desde template '{template_name}' con trace_id: {trace_id}")
        
        # Generar nombre de campaña
        campaign_name = self._generate_campaign_name(template_name, config, custom_data)
        
        # Obtener datos de productos desde ANCHOR
        products_data = await self.anchor_client.get_products_data(
            config.market, 
            config.product_line
        )
        
        # Crear grupos de anuncios
        ad_groups = await self._create_ad_groups_from_template(
            template, 
            config, 
            products_data, 
            custom_data
        )
        
        # Calcular métricas estimadas
        metrics = self._calculate_campaign_metrics(config, len(ad_groups))
        
        # Crear campaña
        campaign = Campaign(
            campaign_id=str(uuid.uuid4()),
            config=config,
            ad_groups=ad_groups,
            total_budget=config.budget,
            estimated_reach=metrics["estimated_reach"],
            estimated_clicks=metrics["estimated_clicks"],
            estimated_conversions=metrics["estimated_conversions"],
            created_at=datetime.datetime.utcnow(),
            trace_id=trace_id,
            status="draft"
        )
        
        logger.info(f"Campaña creada exitosamente: {campaign.campaign_id}")
        return campaign
    
    def _generate_trace_id(self, config: CampaignConfig) -> str:
        """Genera un trace_id único para auditoría"""
        trace_data = f"{config.cmd_id}-{config.market}-{datetime.datetime.utcnow().isoformat()}"
        return hashlib.sha256(trace_data.encode()).hexdigest()[:16]
    
    def _generate_campaign_name(self, template_name: str, config: CampaignConfig, custom_data: Dict = None) -> str:
        """Genera nombre descriptivo de campaña"""
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        market_name = config.market.upper()
        
        if custom_data and "product_name" in custom_data:
            return f"{market_name}_{template_name}_{custom_data['product_name']}_{current_date}"
        else:
            return f"{market_name}_{template_name}_{config.product_line}_{current_date}"
    
    async def _create_ad_groups_from_template(
        self, 
        template: Dict, 
        config: CampaignConfig, 
        products_data: List[Dict],
        custom_data: Optional[Dict] = None
    ) -> List[AdGroup]:
        """Crea grupos de anuncios basados en el template"""
        ad_groups = []
        
        # Crear grupos por cada producto o categoría
        for i, product in enumerate(products_data[:5]):  # Máximo 5 productos
            ad_group_id = f"{config.cmd_id}_group_{i+1}"
            group_name = f"{template['name']} - {product.get('name', f'Product {i+1}')}"
            
            # Generar keywords automáticamente
            keywords = self._generate_keywords_for_product(product, template, config)
            
            # Crear creativos usando AdCopyGenerator
            ad_copy_gen = AdCopyGenerator()
            creatives = await ad_copy_gen.generate_creatives(
                template, 
                product, 
                config, 
                num_variants=3 if config.ab_testing else 1
            )
            
            # Calcular presupuesto por grupo
            group_budget = config.budget / len(products_data[:5])
            
            ad_group = AdGroup(
                ad_group_id=ad_group_id,
                name=group_name,
                keywords=keywords,
                creatives=creatives,
                daily_budget=group_budget / config.duration_days,
                bid_strategy=template["bid_strategy"]
            )
            
            ad_groups.append(ad_group)
        
        return ad_groups
    
    def _generate_keywords_for_product(self, product: Dict, template: Dict, config: CampaignConfig) -> List[str]:
        """Genera keywords automáticamente para un producto"""
        keywords = []
        product_name = product.get("name", "").lower()
        category = product.get("category", "").lower()
        
        # Keywords base del producto
        if product_name:
            keywords.extend([
                f'"{product_name}"',  # Exact match
                product_name,         # Broad match
                f'{product_name} buy',
                f'{product_name} price',
                f'best {product_name}'
            ])
        
        # Keywords por categoría
        if category:
            keywords.extend([
                category,
                f'{category} online',
                f'buy {category}',
                f'{category} deals',
                f'best {category}'
            ])
        
        # Keywords de intención
        intent_keywords = [
            "buy", "purchase", "order", "shop", "discount", "sale", "offer",
            "free shipping", "best price", "compare", "review", "rating"
        ]
        
        for intent in intent_keywords:
            if category:
                keywords.append(f"{category} {intent}")
        
        return keywords[:20]  # Máximo 20 keywords por grupo
    
    def _calculate_campaign_metrics(self, config: CampaignConfig, num_ad_groups: int) -> Dict[str, int]:
        """Calcula métricas estimadas para la campaña"""
        # Cálculos basados en benchmarks de la industria
        base_ctr = 0.02  # 2% CTR promedio
        base_cvr = 0.05  # 5% tasa de conversión promedio
        
        # Ajustes por objetivo
        objective_multipliers = {
            CampaignObjective.AWARENESS: {"ctr": 1.5, "cvr": 0.5},
            CampaignObjective.TRAFFIC: {"ctr": 1.2, "cvr": 0.8},
            CampaignObjective.ENGAGEMENT: {"ctr": 1.3, "cvr": 0.6},
            CampaignObjective.LEAD_GENERATION: {"ctr": 1.1, "cvr": 1.2},
            CampaignObjective.CONVERSIONS: {"ctr": 0.9, "cvr": 1.5},
            CampaignObjective.SALES: {"ctr": 0.8, "cvr": 1.8}
        }
        
        multiplier = objective_multipliers.get(config.objective, {"ctr": 1.0, "cvr": 1.0})
        
        # Estimación de reach (alcanzabilidad promedio)
        estimated_reach = int(config.budget * 50)  # $1 = ~50 personas alcanzadas
        
        # Estimación de clicks
        estimated_clicks = int(estimated_reach * base_ctr * multiplier["ctr"])
        
        # Estimación de conversiones
        estimated_conversions = int(estimated_clicks * base_cvr * multiplier["cvr"])
        
        return {
            "estimated_reach": estimated_reach,
            "estimated_clicks": estimated_clicks,
            "estimated_conversions": estimated_conversions
        }


class TargetingEngine:
    """Motor de targeting automático con geolocalización, demografía e intereses"""
    
    def __init__(self):
        self.geo_data = self._load_geo_data()
        self.demographic_profiles = self._load_demographic_profiles()
        self.interest_categories = self._load_interest_categories()
        
    def _load_geo_data(self) -> Dict[str, Any]:
        """Carga datos geográficos para targeting"""
        return {
            "US": {
                "regions": ["Northeast", "South", "Midwest", "West"],
                "top_cities": [
                    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                    "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"
                ],
                "timezones": ["EST", "CST", "MST", "PST"],
                "languages": ["en"]
            },
            "MX": {
                "regions": ["Norte", "Centro", "Sur", "Golfo", "Península"],
                "top_cities": [
                    "Ciudad de México", "Guadalajara", "Monterrey", "Puebla", "Tijuana",
                    "León", "Juárez", "Torreón", "Querétaro", "Mérida"
                ],
                "timezones": ["CST"],
                "languages": ["es"]
            },
            "ES": {
                "regions": ["Norte", "Sur", "Este", "Oeste", "Centro"],
                "top_cities": [
                    "Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza",
                    "Málaga", "Murcia", "Palma", "Las Palmas", "Bilbao"
                ],
                "timezones": ["CET"],
                "languages": ["es", "ca", "eu", "gl"]
            }
        }
    
    def _load_demographic_profiles(self) -> Dict[str, Any]:
        """Carga perfiles demográficos predefinidos"""
        return {
            "young_professionals": {
                "age_range": (22, 32),
                "gender": "all",
                "interests": ["Career", "Technology", "Networking", "Personal Development"],
                "behaviors": ["Early Adopters", "Mobile Users", "Social Media Users"],
                "income_level": "middle_to_high"
            },
            "families": {
                "age_range": (28, 45),
                "gender": "all",
                "interests": ["Family", "Parenting", "Education", "Home Improvement"],
                "behaviors": ["Frequent Shoppers", "Value Seekers"],
                "income_level": "middle"
            },
            "seniors": {
                "age_range": (55, 75),
                "gender": "all",
                "interests": ["Health", "Travel", "Finance", "Hobbies"],
                "behaviors": ["Brand Loyal", "Research Before Purchase"],
                "income_level": "varied"
            },
            "students": {
                "age_range": (18, 25),
                "gender": "all",
                "interests": ["Education", "Entertainment", "Technology", "Fashion"],
                "behaviors": ["Price Sensitive", "Social Media Active"],
                "income_level": "low"
            },
            "entrepreneurs": {
                "age_range": (25, 50),
                "gender": "all",
                "interests": ["Business", "Leadership", "Innovation", "Technology"],
                "behaviors": ["Decision Makers", "High Value Customers"],
                "income_level": "high"
            }
        }
    
    def _load_interest_categories(self) -> Dict[str, List[str]]:
        """Carga categorías de intereses organizadas"""
        return {
            "technology": [
                "Artificial Intelligence", "Machine Learning", "Robotics", "Mobile Apps",
                "Software Development", "Cloud Computing", "Cybersecurity", "IoT"
            ],
            "business": [
                "Entrepreneurship", "Startups", "Marketing", "Sales", "Leadership",
                "Management", "Finance", "Investments", "Real Estate", "Consulting"
            ],
            "lifestyle": [
                "Health & Fitness", "Travel", "Food & Cooking", "Fashion", "Beauty",
                "Home & Garden", "Parenting", "Relationships", "Hobbies", "Sports"
            ],
            "entertainment": [
                "Movies", "Music", "Gaming", "TV Shows", "Books", "Comedy",
                "Concerts", "Festivals", "Celebrities", "Comics"
            ],
            "education": [
                "Online Learning", "Professional Development", "Languages", "Science",
                "History", "Art", "Mathematics", "Literature", "Research", "Books"
            ],
            "shopping": [
                "E-commerce", "Fashion", "Electronics", "Home Improvement", "Sports",
                "Books", "Groceries", "Beauty Products", "Gifts", "Deals & Discounts"
            ]
        }
    
    async def generate_targeting_criteria(
        self,
        campaign_config: CampaignConfig,
        product_data: Dict,
        target_persona: Optional[str] = None
    ) -> TargetingCriteria:
        """Genera criterios de targeting automático basados en producto y audiencia"""
        
        logger.info(f"Generando targeting para campaña {campaign_config.cmd_id}")
        
        # Targeting geográfico
        geo_targeting = self._generate_geo_targeting(campaign_config.market)
        
        # Targeting demográfico
        demo_targeting = self._generate_demographic_targeting(
            campaign_config.objective, 
            target_persona,
            product_data
        )
        
        # Targeting por intereses
        interest_targeting = self._generate_interest_targeting(product_data)
        
        # Targeting por comportamientos
        behavior_targeting = self._generate_behavior_targeting(campaign_config.objective)
        
        # Combinar todos los criterios
        targeting = TargetingCriteria(
            countries=[campaign_config.market],
            **geo_targeting,
            **demo_targeting,
            interests=interest_targeting,
            behaviors=behavior_targeting,
            languages=geo_targeting.get("languages", ["en"])
        )
        
        logger.info(f"Targeting generado: {len(targeting.countries)} países, "
                   f"{len(targeting.interests)} intereses, "
                   f"{targeting.age_range[0]}-{targeting.age_range[1]} años")
        
        return targeting
    
    def _generate_geo_targeting(self, market: str) -> Dict[str, Any]:
        """Genera targeting geográfico basado en el mercado"""
        geo_data = self.geo_data.get(market, {})
        
        return {
            "regions": geo_data.get("regions", []),
            "cities": geo_data.get("top_cities", []),
            "languages": geo_data.get("languages", ["en"])
        }
    
    def _generate_demographic_targeting(
        self,
        objective: CampaignObjective,
        target_persona: Optional[str],
        product_data: Dict
    ) -> Dict[str, Any]:
        """Genera targeting demográfico"""
        
        # Si se especifica una persona objetivo, usar su perfil
        if target_persona and target_persona in self.demographic_profiles:
            profile = self.demographic_profiles[target_persona]
            return {
                "age_range": profile["age_range"],
                "gender": profile["gender"]
            }
        
        # Ajustes por objetivo de campaña
        demographic_adjustments = {
            CampaignObjective.AWARENESS: {"age_range": (18, 65), "gender": "all"},
            CampaignObjective.TRAFFIC: {"age_range": (18, 54), "gender": "all"},
            CampaignObjective.ENGAGEMENT: {"age_range": (18, 44), "gender": "all"},
            CampaignObjective.LEAD_GENERATION: {"age_range": (25, 55), "gender": "all"},
            CampaignObjective.CONVERSIONS: {"age_range": (25, 54), "gender": "all"},
            CampaignObjective.SALES: {"age_range": (25, 64), "gender": "all"}
        }
        
        return demographic_adjustments.get(objective, {"age_range": (18, 54), "gender": "all"})
    
    def _generate_interest_targeting(self, product_data: Dict) -> List[str]:
        """Genera targeting por intereses basado en el producto"""
        interests = []
        category = product_data.get("category", "").lower()
        name = product_data.get("name", "").lower()
        
        # Mapear categoría del producto a intereses
        category_mapping = {
            "technology": "technology",
            "software": "technology",
            "business": "business",
            "service": "business",
            "fashion": "lifestyle",
            "clothing": "lifestyle",
            "food": "lifestyle",
            "restaurant": "lifestyle",
            "game": "entertainment",
            "book": "entertainment",
            "course": "education",
            "training": "education",
            "product": "shopping"
        }
        
        for cat_key, interest_key in category_mapping.items():
            if cat_key in category or cat_key in name:
                interests.extend(self.interest_categories.get(interest_key, []))
        
        # Si no hay intereses específicos, usar intereses generales
        if not interests:
            interests = [
                "Shopping", "Online Shopping", "Technology", "Business",
                "Entertainment", "Health", "Travel", "Education"
            ]
        
        return list(set(interests))[:10]  # Máximo 10 intereses
    
    def _generate_behavior_targeting(self, objective: CampaignObjective) -> List[str]:
        """Genera targeting por comportamientos"""
        behavior_mapping = {
            CampaignObjective.AWARENESS: ["Online", "Social Media Users"],
            CampaignObjective.TRAFFIC: ["Website Visitors", "Online Shoppers"],
            CampaignObjective.ENGAGEMENT: ["Social Media Users", "Content Consumers"],
            CampaignObjective.LEAD_GENERATION: ["Research Users", "Lead Prospects"],
            CampaignObjective.CONVERSIONS: ["Online Buyers", "Purchase Intent"],
            CampaignObjective.SALES: ["Frequent Buyers", "High Value Customers"]
        }
        
        return behavior_mapping.get(objective, ["Online Users"])


class AdCopyGenerator:
    """Generador de ad copy usando AI/LLM"""
    
    def __init__(self, gemini_api_key: str = None):
        self.gemini_api_key = gemini_api_key
        self.copy_templates = self._load_copy_templates()
        self.cta_variations = self._load_cta_variations()
        self.emotional_triggers = self._load_emotional_triggers()
        
    def _load_copy_templates(self) -> Dict[str, List[str]]:
        """Carga templates de copy predefinidos"""
        return {
            "headlines": {
                "product_launch": [
                    "Nuevo {product}: {benefit}",
                    "Descubre {product}",
                    "Presentación exclusiva: {product}",
                    "¡Ya disponible! {product}",
                    "Innovación en {category}: {product}"
                ],
                "promotional": [
                    "{discount}% OFF - {product}",
                    "¡Oferta limitada! {discount}% de descuento",
                    "Black Friday: {product} con {discount}% OFF",
                    "Liquidación {product} - {discount}% OFF",
                    "Precio especial: {product} por solo ${price}"
                ],
                "urgency": [
                    "Solo hasta {date}: {product}",
                    "Última oportunidad: {product}",
                    "No te lo pierdas: {product}",
                    "Quedan pocas horas: {product}",
                    "Venta termina pronto: {product}"
                ],
                "benefit_focused": [
                    "{product}: {main_benefit}",
                    "Obtén {main_benefit} con {product}",
                    "La solución que buscabas: {product}",
                    "{product} para {specific_benefit}",
                    "Transforma tu {area} con {product}"
                ]
            },
            "descriptions": {
                "feature_oriented": [
                    "{product} ofrece {main_feature}. Perfecto para {target_audience}.",
                    "Descubre {main_feature} con {product}. {additional_benefit}",
                    "{product} combina {feature1} y {feature2} para {main_benefit}.",
                    "Experimenta {main_feature}. {product} es la elección ideal para {use_case}.",
                    "{product} redefine {category} con {main_feature}."
                ],
                "problem_solution": [
                    "¿Problema con {problem}? {product} es la solución.",
                    "Elimina {problem} para siempre con {product}.",
                    "{product} resuelve {problem} de manera {solution_method}.",
                    "Adiós {problem}. Hola {product}.",
                    "Deja de {problem}. Comienza con {product}."
                ],
                "social_proof": [
                    "Más de {number}+ {positive_review} confían en {product}.",
                    "Voted #1 in {category} by {review_source}.",
                    "{number}+ clientes satisfechos no pueden estar equivocados.",
                    "La elección de {number}+ profesionales: {product}.",
                    "Rated {rating}/5 stars by {number}+ users."
                ]
            }
        }
    
    def _load_cta_variations(self) -> Dict[str, List[str]]:
        """Carga variaciones de call-to-action"""
        return {
            "awareness": ["Conoce más", "Descubre", "Explora", "Ver detalles"],
            "traffic": ["Visitar sitio", "Ir al sitio", "Conocer más", "Ver más"],
            "engagement": ["Me gusta", "Compartir", "Comentar", "Seguir"],
            "lead_generation": ["Descargar", "Registrarse", "Solicitar", "Obtener"],
            "conversions": ["Comprar ahora", "Ordenar", "Adquirir", "Comprar"],
            "sales": ["Comprar", "Ordenar ya", "Adquirir", "Comprar ahora"]
        }
    
    def _load_emotional_triggers(self) -> List[str]:
        """Carga triggers emocionales para copy persuasivo"""
        return [
            "exclusive", "limited time", "free", "save", "new", "improved",
            "guaranteed", "proven", "trusted", "premium", "professional",
            "easy", "simple", "fast", "instant", "secure", "safe"
        ]
    
    async def generate_creatives(
        self,
        template: Dict,
        product_data: Dict,
        campaign_config: CampaignConfig,
        num_variants: int = 3
    ) -> List[Creative]:
        """Genera múltiples variantes de creativos para A/B testing"""
        creatives = []
        
        for variant in range(num_variants):
            variant_letter = chr(ord('A') + variant)  # A, B, C
            
            # Generar copy usando templates
            copy_data = await self._generate_copy_variants(
                template, 
                product_data, 
                campaign_config, 
                variant_letter
            )
            
            # Crear creativo
            creative = Creative(
                creative_id=f"{campaign_config.cmd_id}_creative_{variant_letter}",
                title=copy_data["title"],
                description=copy_data["description"],
                headline=copy_data["headline"],
                call_to_action=copy_data["cta"],
                final_url=product_data.get("url", "https://example.com"),
                ab_variant=variant_letter
            )
            
            creatives.append(creative)
        
        logger.info(f"Generados {len(creatives)} creativos para A/B testing")
        return creatives
    
    async def _generate_copy_variants(
        self,
        template: Dict,
        product_data: Dict,
        campaign_config: CampaignConfig,
        variant_letter: str
    ) -> Dict[str, str]:
        """Genera una variante específica de copy"""
        
        # Extraer datos del producto
        product_name = product_data.get("name", "Nuestro Producto")
        category = product_data.get("category", "categoria")
        description = product_data.get("description", "Producto de calidad")
        
        # Obtener templates específicos
        creative_templates = template.get("creative_templates", {})
        headline_templates = creative_templates.get("headline", [])
        description_templates = creative_templates.get("description", [])
        cta_templates = creative_templates.get("call_to_action", [])
        
        # Seleccionar template específico para la variante
        headline_idx = (ord(variant_letter) - ord('A')) % len(headline_templates)
        description_idx = (ord(variant_letter) - ord('A')) % len(description_templates)
        cta_idx = (ord(variant_letter) - ord('A')) % len(cta_templates)
        
        # Generar datos dinámicos para reemplazos
        dynamic_data = self._generate_dynamic_data(product_data, campaign_config)
        
        # Rellenar templates
        headline = self._fill_template(
            headline_templates[headline_idx], 
            dynamic_data
        )
        
        description = self._fill_template(
            description_templates[description_idx], 
            dynamic_data
        )
        
        cta = self._fill_template(
            cta_templates[cta_idx], 
            dynamic_data
        )
        
        # Generar título más corto para redes sociales
        title = headline[:30] + "..." if len(headline) > 30 else headline
        
        return {
            "title": title,
            "description": description,
            "headline": headline,
            "cta": cta
        }
    
    def _generate_dynamic_data(self, product_data: Dict, campaign_config: CampaignConfig) -> Dict[str, str]:
        """Genera datos dinámicos para reemplazar en templates"""
        current_date = datetime.datetime.now()
        
        # Datos base del producto
        dynamic_data = {
            "product_name": product_data.get("name", "Producto"),
            "product": product_data.get("name", "Producto"),
            "category": product_data.get("category", "categoria"),
            "category_name": product_data.get("category", "categoria"),
            "product_category": product_data.get("category", "categoria"),
            "description": product_data.get("description", "descripción"),
            "key_benefit": product_data.get("key_benefit", "beneficio principal"),
            "main_benefit": product_data.get("main_benefit", "beneficio principal"),
            "specific_benefit": product_data.get("specific_benefit", "beneficio específico"),
            "price": product_data.get("price", "precio especial"),
            "url": product_data.get("url", "https://example.com")
        }
        
        # Datos específicos por tipo de campaña
        if "black_friday" in campaign_config.intent.lower():
            discount = random.randint(20, 70)
            duration = random.choice(["hoy", "esta semana", "por tiempo limitado"])
            dynamic_data.update({
                "discount": f"{discount}%",
                "duration": duration
            })
        
        if campaign_config.objective == CampaignObjective.LEAD_GENERATION:
            lead_magnets = ["Guía gratuita", "E-book", "Plantilla", "Curso online", "Webinar"]
            dynamic_data.update({
                "lead_magnet": random.choice(lead_magnets),
                "free_item": random.choice(lead_magnets)
            })
        
        # Datos de fecha
        dynamic_data.update({
            "date": (current_date + datetime.timedelta(days=7)).strftime("%d/%m"),
            "current_date": current_date.strftime("%d/%m/%Y")
        })
        
        return dynamic_data
    
    def _fill_template(self, template: str, data: Dict[str, str]) -> str:
        """Rellena un template con datos dinámicos"""
        try:
            return template.format(**data)
        except KeyError as e:
            logger.warning(f"Clave faltante en template: {e}")
            return template  # Devolver template original si hay error
    
    async def generate_ab_test_variants(
        self,
        base_creative: Creative,
        num_variants: int = 5
    ) -> List[Creative]:
        """Genera variantes para A/B testing de un creativo base"""
        variants = []
        
        # Variante original
        original = Creative(
            creative_id=f"{base_creative.creative_id}_ORIGINAL",
            title=base_creative.title,
            description=base_creative.description,
            headline=base_creative.headline,
            call_to_action=base_creative.call_to_action,
            final_url=base_creative.final_url,
            ab_variant="ORIGINAL"
        )
        variants.append(original)
        
        # Variantes modificadas
        test_strategies = [
            {"modification": "headline", "focus": "benefit"},
            {"modification": "headline", "focus": "urgency"},
            {"modification": "description", "focus": "social_proof"},
            {"modification": "cta", "focus": "action"},
            {"modification": "description", "focus": "features"}
        ]
        
        for i in range(min(num_variants - 1, len(test_strategies))):
            strategy = test_strategies[i]
            variant = await self._create_variant(base_creative, strategy, i + 1)
            variants.append(variant)
        
        logger.info(f"Generadas {len(variants)} variantes para A/B testing")
        return variants
    
    async def _create_variant(
        self,
        base_creative: Creative,
        strategy: Dict[str, str],
        variant_num: int
    ) -> Creative:
        """Crea una variante específica basada en estrategia de testing"""
        modification = strategy["modification"]
        focus = strategy["focus"]
        
        # Copiar datos base
        new_title = base_creative.title
        new_description = base_creative.description
        new_headline = base_creative.headline
        new_cta = base_creative.call_to_action
        
        # Aplicar modificaciones según estrategia
        if modification == "headline" and focus == "benefit":
            # Cambiar headline enfocándose en beneficios
            new_headline = f"Transforma tu {focus} con nuestro producto"
            new_title = new_headline[:30] + "..." if len(new_headline) > 30 else new_headline
            
        elif modification == "headline" and focus == "urgency":
            # Agregar urgencia al headline
            new_headline = f"¡Oferta limitada! {base_creative.headline}"
            new_title = new_headline[:30] + "..." if len(new_headline) > 30 else new_headline
            
        elif modification == "description" and focus == "social_proof":
            # Agregar prueba social
            new_description = f"{base_creative.description} Más de 1000+ clientes satisfechos."
            
        elif modification == "cta" and focus == "action":
            # Cambiar call-to-action para más acción
            action_ctas = ["¡Comprar ahora!", "¡Ordenar ya!", "¡Adquirir ahora!"]
            new_cta = random.choice(action_ctas)
            
        elif modification == "description" and focus == "features":
            # Enfocar en características
            new_description = f"{base_creative.description} Características premium incluidas."
        
        return Creative(
            creative_id=f"{base_creative.creative_id}_VARIANT_{variant_num}",
            title=new_title,
            description=new_description,
            headline=new_headline,
            call_to_action=new_cta,
            final_url=base_creative.final_url,
            ab_variant=f"VARIANT_{variant_num}"
        )


class BudgetCalculator:
    """Calculadora automática de presupuestos por objetivo"""
    
    def __init__(self):
        self.benchmarks = self._load_benchmarks()
        self.platform_costs = self._load_platform_costs()
    
    def _load_benchmarks(self) -> Dict[str, Any]:
        """Carga benchmarks de la industria para cálculo de presupuestos"""
        return {
            CampaignObjective.AWARENESS: {
                "avg_cpm": 5.0,  # Costo por mil impresiones
                "avg_ctr": 0.015,  # Click-through rate
                "avg_cvr": 0.01,   # Conversion rate
                "budget_distribution": {"display": 0.6, "video": 0.4}
            },
            CampaignObjective.TRAFFIC: {
                "avg_cpc": 1.50,  # Costo por clic
                "avg_ctr": 0.025,
                "avg_cvr": 0.02,
                "budget_distribution": {"search": 0.7, "display": 0.3}
            },
            CampaignObjective.ENGAGEMENT: {
                "avg_cpe": 0.10,  # Costo por engagement
                "avg_engagement_rate": 0.035,
                "budget_distribution": {"social": 0.8, "display": 0.2}
            },
            CampaignObjective.LEAD_GENERATION: {
                "avg_cpl": 25.0,  # Costo por lead
                "avg_ctr": 0.030,
                "avg_cvr": 0.08,
                "budget_distribution": {"search": 0.6, "social": 0.4}
            },
            CampaignObjective.CONVERSIONS: {
                "avg_cpa": 75.0,  # Costo por adquisición
                "avg_ctr": 0.035,
                "avg_cvr": 0.12,
                "budget_distribution": {"search": 0.7, "shopping": 0.3}
            },
            CampaignObjective.SALES: {
                "avg_cpa": 100.0,
                "avg_ctr": 0.040,
                "avg_cvr": 0.15,
                "budget_distribution": {"search": 0.5, "shopping": 0.3, "display": 0.2}
            }
        }
    
    def _load_platform_costs(self) -> Dict[str, float]:
        """Carga costos promedio por plataforma"""
        return {
            CampaignPlatform.GOOGLE_ADS: 1.0,      # Base cost
            CampaignPlatform.META_ADS: 0.8,        # 20% más barato
            CampaignPlatform.LINKEDIN_ADS: 2.5,    # 150% más caro
            CampaignPlatform.TIKTOK_ADS: 0.6,      # 40% más barato
            CampaignPlatform.TWITTER_ADS: 1.2      # 20% más caro
        }
    
    def calculate_budget_allocation(
        self,
        campaign_config: CampaignConfig,
        target_audience_size: int
    ) -> Dict[str, Any]:
        """Calcula distribución de presupuesto para la campaña"""
        
        objective = campaign_config.objective
        platforms = campaign_config.platform
        total_budget = campaign_config.budget
        duration_days = campaign_config.duration_days
        
        if objective not in self.benchmarks:
            raise ValueError(f"Objetivo '{objective}' no encontrado en benchmarks")
        
        benchmark = self.benchmarks[objective]
        
        # Calcular presupuesto por plataforma
        platform_budgets = self._allocate_budget_by_platform(
            total_budget, 
            platforms, 
            benchmark.get("budget_distribution", {})
        )
        
        # Calcular presupuesto diario
        daily_budget = total_budget / duration_days
        
        # Calcular métricas estimadas
        estimated_metrics = self._calculate_estimated_metrics(
            total_budget, 
            benchmark, 
            target_audience_size
        )
        
        # Calcular bid strategies
        bid_strategies = self._calculate_bid_strategies(
            platforms, 
            benchmark
        )
        
        budget_plan = {
            "total_budget": total_budget,
            "daily_budget": daily_budget,
            "duration_days": duration_days,
            "platform_budgets": platform_budgets,
            "estimated_metrics": estimated_metrics,
            "bid_strategies": bid_strategies,
            "objective": objective.value,
            "benchmark_used": benchmark
        }
        
        logger.info(f"Budget allocation calculado: ${total_budget} "
                   f"para {duration_days} días, "
                   f"${daily_budget:.2f} diario")
        
        return budget_plan
    
    def _allocate_budget_by_platform(
        self,
        total_budget: float,
        platforms: List[CampaignPlatform],
        distribution_template: Dict[str, float]
    ) -> Dict[str, float]:
        """Distribuye presupuesto entre plataformas"""
        platform_budgets = {}
        
        if len(platforms) == 1:
            # Solo una plataforma
            platform_budgets[platforms[0].value] = total_budget
        else:
            # Múltiples plataformas
            total_weight = sum(self.platform_costs.get(p, 1.0) for p in platforms)
            
            for platform in platforms:
                # Ajustar peso por costo de plataforma
                adjusted_weight = 1.0 / self.platform_costs.get(platform, 1.0)
                budget_share = (adjusted_weight / total_weight) * total_budget
                platform_budgets[platform.value] = round(budget_share, 2)
        
        return platform_budgets
    
    def _calculate_estimated_metrics(
        self,
        total_budget: float,
        benchmark: Dict[str, Any],
        audience_size: int
    ) -> Dict[str, int]:
        """Calcula métricas estimadas basadas en benchmarks"""
        
        # Métricas base
        if "avg_cpm" in benchmark:
            impressions = int((total_budget / benchmark["avg_cpm"]) * 1000)
            clicks = int(impressions * benchmark["avg_ctr"])
            conversions = int(clicks * benchmark["avg_cvr"])
        elif "avg_cpc" in benchmark:
            clicks = int(total_budget / benchmark["avg_cpc"])
            impressions = int(clicks / benchmark["avg_ctr"])
            conversions = int(clicks * benchmark["avg_cvr"])
        elif "avg_cpl" in benchmark:
            leads = int(total_budget / benchmark["avg_cpl"])
            clicks = int(leads / benchmark["avg_cvr"])
            impressions = int(clicks / benchmark["avg_ctr"])
            conversions = leads
        elif "avg_cpa" in benchmark:
            conversions = int(total_budget / benchmark["avg_cpa"])
            clicks = int(conversions / benchmark["avg_cvr"])
            impressions = int(clicks / benchmark["avg_ctr"])
        else:
            # Cálculo genérico
            impressions = audience_size
            clicks = int(impressions * 0.02)  # 2% CTR promedio
            conversions = int(clicks * 0.05)  # 5% CVR promedio
        
        return {
            "estimated_impressions": impressions,
            "estimated_clicks": clicks,
            "estimated_conversions": conversions
        }
    
    def _calculate_bid_strategies(
        self,
        platforms: List[CampaignPlatform],
        benchmark: Dict[str, Any]
    ) -> Dict[str, str]:
        """Calcula estrategias de bidding para cada plataforma"""
        strategies = {}
        
        for platform in platforms:
            if platform == CampaignPlatform.GOOGLE_ADS:
                if "avg_cpc" in benchmark:
                    strategies[platform.value] = "maximize_clicks"
                elif "avg_cpm" in benchmark:
                    strategies[platform.value] = "target_impressions"
                else:
                    strategies[platform.value] = "maximize_conversions"
            
            elif platform == CampaignPlatform.META_ADS:
                if benchmark.get("avg_engagement_rate"):
                    strategies[platform.value] = "engagement"
                else:
                    strategies[platform.value] = "reach"
            
            else:
                # Estrategia por defecto
                strategies[platform.value] = "automatic"
        
        return strategies


class CampaignScheduler:
    """Sistema de programación de campañas (fechas, horarios)"""
    
    def __init__(self):
        self.timezone_data = self._load_timezone_data()
        self.optimal_schedule = self._load_optimal_schedule()
    
    def _load_timezone_data(self) -> Dict[str, str]:
        """Carga datos de zonas horarias por mercado"""
        return {
            "US": "America/New_York",
            "MX": "America/Mexico_City", 
            "ES": "Europe/Madrid",
            "UK": "Europe/London",
            "BR": "America/Sao_Paulo",
            "AR": "America/Argentina/Buenos_Aires"
        }
    
    def _load_optimal_schedule(self) -> Dict[str, Dict]:
        """Carga horarios óptimos por tipo de audiencia y plataforma"""
        return {
            "b2b": {
                "weekdays": {
                    "start_hour": 9,  # 9 AM
                    "end_hour": 17,   # 5 PM
                    "optimal_hours": [9, 10, 12, 13, 16, 17]
                },
                "weekends": {
                    "start_hour": 10,
                    "end_hour": 14,
                    "optimal_hours": [10, 11, 12, 13]
                }
            },
            "b2c": {
                "weekdays": {
                    "start_hour": 19,  # 7 PM
                    "end_hour": 23,    # 11 PM
                    "optimal_hours": [19, 20, 21, 22]
                },
                "weekends": {
                    "start_hour": 10,
                    "end_hour": 23,
                    "optimal_hours": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                }
            },
            "ecommerce": {
                "weekdays": {
                    "start_hour": 20,  # 8 PM
                    "end_hour": 23,    # 11 PM
                    "optimal_hours": [20, 21, 22]
                },
                "weekends": {
                    "start_hour": 9,
                    "end_hour": 22,
                    "optimal_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
                }
            }
        }
    
    def generate_campaign_schedule(
        self,
        campaign_config: CampaignConfig,
        target_audience_type: str = "b2c"
    ) -> Dict[str, Any]:
        """Genera programación óptima de campaña"""
        
        market = campaign_config.market
        start_date = campaign_config.start_date or datetime.datetime.now()
        end_date = campaign_config.end_date or (start_date + datetime.timedelta(days=campaign_config.duration_days))
        
        # Obtener zona horaria del mercado
        timezone = self.timezone_data.get(market, "UTC")
        
        # Obtener configuración de horarios óptimos
        schedule_config = self.optimal_schedule.get(target_audience_type, self.optimal_schedule["b2c"])
        
        # Generar programación detallada
        schedule = self._create_detailed_schedule(
            start_date, 
            end_date, 
            schedule_config, 
            timezone
        )
        
        # Calcular distribución de presupuesto por horario
        budget_distribution = self._calculate_budget_distribution(schedule, campaign_config.budget)
        
        schedule_plan = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timezone": timezone,
            "audience_type": target_audience_type,
            "schedule": schedule,
            "budget_distribution": budget_distribution,
            "total_hours_active": len(schedule["active_hours"]),
            "optimization_notes": self._generate_optimization_notes(schedule, target_audience_type)
        }
        
        logger.info(f"Programación generada: {len(schedule['active_hours'])} horas activas "
                   f"en zona horaria {timezone}")
        
        return schedule_plan
    
    def _create_detailed_schedule(
        self,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        schedule_config: Dict,
        timezone: str
    ) -> Dict[str, Any]:
        """Crea programación detallada día por día"""
        schedule = {
            "daily_schedule": {},
            "active_hours": [],
            "peak_hours": []
        }
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            is_weekend = current_date.weekday() >= 5  # Sábado=5, Domingo=6
            
            if is_weekend:
                daily_config = schedule_config.get("weekends", schedule_config["weekdays"])
            else:
                daily_config = schedule_config["weekdays"]
            
            # Generar horas activas para el día
            active_hours = list(range(daily_config["start_hour"], daily_config["end_hour"] + 1))
            optimal_hours = daily_config.get("optimal_hours", active_hours)
            
            schedule["daily_schedule"][date_str] = {
                "active_hours": active_hours,
                "optimal_hours": optimal_hours,
                "budget_multiplier": 1.2 if any(h in optimal_hours for h in active_hours) else 1.0
            }
            
            # Agregar a listas globales
            for hour in active_hours:
                schedule["active_hours"].append(f"{date_str}T{hour:02d}:00:00")
            
            for hour in optimal_hours:
                if hour in active_hours:
                    schedule["peak_hours"].append(f"{date_str}T{hour:02d}:00:00")
            
            current_date += datetime.timedelta(days=1)
        
        return schedule
    
    def _calculate_budget_distribution(
        self,
        schedule: Dict,
        total_budget: float
    ) -> Dict[str, float]:
        """Calcula distribución de presupuesto por horario"""
        distribution = {}
        
        # Presupuesto base por hora
        total_active_hours = len(schedule["active_hours"])
        base_budget_per_hour = total_budget / total_active_hours
        
        # Multiplicadores para horas pico
        peak_multiplier = 1.3
        normal_multiplier = 1.0
        
        for hour in schedule["active_hours"]:
            if hour in schedule["peak_hours"]:
                budget_share = base_budget_per_hour * peak_multiplier
            else:
                budget_share = base_budget_per_hour * normal_multiplier
            
            distribution[hour] = round(budget_share, 2)
        
        return distribution
    
    def _generate_optimization_notes(self, schedule: Dict, audience_type: str) -> List[str]:
        """Genera notas de optimización para la programación"""
        notes = []
        
        peak_hours_count = len(schedule["peak_hours"])
        total_hours_count = len(schedule["active_hours"])
        peak_percentage = (peak_hours_count / total_hours_count) * 100 if total_hours_count > 0 else 0
        
        notes.append(f"Horas pico representan {peak_percentage:.1f}% del tiempo total activo")
        
        if audience_type == "b2b":
            notes.append("Programación optimizada para horarios laborales (9 AM - 5 PM)")
        elif audience_type == "b2c":
            notes.append("Programación optimizada para horarios de ocio (7 PM - 11 PM)")
        elif audience_type == "ecommerce":
            notes.append("Programación optimizada para shopping online (principalmente noches y fines de semana)")
        
        if peak_percentage > 60:
            notes.append("Alta concentración en horas pico - considerar ampliar horarios para reducir competencia")
        elif peak_percentage < 30:
            notes.append("Baja concentración en horas pico - puede ser necesario ajustar para mayor impacto")
        
        return notes


class ABTestManager:
    """Manager de A/B testing automático de creativos"""
    
    def __init__(self):
        self.test_configurations = self._load_test_configurations()
        self.evaluation_metrics = self._load_evaluation_metrics()
    
    def _load_test_configurations(self) -> Dict[str, Any]:
        """Carga configuraciones de A/B testing"""
        return {
            "creative_testing": {
                "min_sample_size": 1000,
                "test_duration_days": 7,
                "significance_level": 0.05,
                "statistical_power": 0.8,
                "metrics": ["ctr", "cvr", "cpc", "cpa"]
            },
            "audience_testing": {
                "min_sample_size": 2000,
                "test_duration_days": 10,
                "significance_level": 0.05,
                "statistical_power": 0.8,
                "metrics": ["reach", "engagement", "cpm"]
            },
            "bidding_testing": {
                "min_sample_size": 500,
                "test_duration_days": 5,
                "significance_level": 0.05,
                "statistical_power": 0.8,
                "metrics": ["cpc", "cpa", "conversion_volume"]
            },
            "placement_testing": {
                "min_sample_size": 1500,
                "test_duration_days": 7,
                "significance_level": 0.05,
                "statistical_power": 0.8,
                "metrics": ["ctr", "viewability", "cpm"]
            }
        }
    
    def _load_evaluation_metrics(self) -> Dict[str, Dict]:
        """Carga métricas de evaluación para A/B tests"""
        return {
            "ctr": {
                "description": "Click-through Rate",
                "calculation": "clicks / impressions",
                "weight": 0.3,
                "higher_is_better": True
            },
            "cvr": {
                "description": "Conversion Rate", 
                "calculation": "conversions / clicks",
                "weight": 0.4,
                "higher_is_better": True
            },
            "cpc": {
                "description": "Cost per Click",
                "calculation": "spend / clicks",
                "weight": 0.2,
                "higher_is_better": False
            },
            "cpa": {
                "description": "Cost per Acquisition",
                "calculation": "spend / conversions",
                "weight": 0.3,
                "higher_is_better": False
            },
            "engagement_rate": {
                "description": "Engagement Rate",
                "calculation": "engagements / impressions",
                "weight": 0.3,
                "higher_is_better": True
            }
        }
    
    def create_ab_test_plan(
        self,
        campaign: Campaign,
        test_type: str = "creative_testing",
        num_variants: int = 3
    ) -> Dict[str, Any]:
        """Crea plan de A/B testing para la campaña"""
        
        if test_type not in self.test_configurations:
            raise ValueError(f"Tipo de test '{test_type}' no encontrado")
        
        config = self.test_configurations[test_type]
        
        # Generar variantes para el test
        variants = self._generate_test_variants(campaign, test_type, num_variants)
        
        # Calcular presupuesto para el test
        test_budget = self._calculate_test_budget(campaign, len(variants), config)
        
        # Crear estructura del test
        test_plan = {
            "test_id": f"ab_test_{campaign.campaign_id}_{test_type}",
            "test_type": test_type,
            "campaign_id": campaign.campaign_id,
            "variants": variants,
            "test_configuration": config,
            "test_budget": test_budget,
            "start_date": datetime.datetime.now().isoformat(),
            "end_date": (datetime.datetime.now() + datetime.timedelta(days=config["test_duration_days"])).isoformat(),
            "evaluation_criteria": self._generate_evaluation_criteria(config),
            "success_threshold": self._calculate_success_threshold(config),
            "auto_optimization": True
        }
        
        logger.info(f"Plan de A/B test creado: {test_type} con {len(variants)} variantes")
        return test_plan
    
    def _generate_test_variants(
        self,
        campaign: Campaign,
        test_type: str,
        num_variants: int
    ) -> List[Dict]:
        """Genera variantes para el A/B test"""
        variants = []
        
        for i in range(num_variants):
            variant_id = f"variant_{chr(ord('A') + i)}"
            variant_name = f"Variante {chr(ord('A') + i)}"
            
            if test_type == "creative_testing":
                variant = self._create_creative_variant(variant_id, variant_name, i, campaign)
            elif test_type == "audience_testing":
                variant = self._create_audience_variant(variant_id, variant_name, i, campaign)
            elif test_type == "bidding_testing":
                variant = self._create_bidding_variant(variant_id, variant_name, i, campaign)
            else:
                variant = self._create_generic_variant(variant_id, variant_name, i, campaign)
            
            variants.append(variant)
        
        return variants
    
    def _create_creative_variant(
        self,
        variant_id: str,
        variant_name: str,
        index: int,
        campaign: Campaign
    ) -> Dict[str, Any]:
        """Crea variante para testing de creativos"""
        creative_approaches = [
            "headline_focused",
            "benefit_focused", 
            "urgency_focused",
            "social_proof_focused",
            "feature_focused"
        ]
        
        approach = creative_approaches[index % len(creative_approaches)]
        
        return {
            "variant_id": variant_id,
            "variant_name": variant_name,
            "approach": approach,
            "description": f"Testing enfoque: {approach.replace('_', ' ').title()}",
            "creative_elements": {
                "headline_style": approach,
                "description_style": approach,
                "visual_style": "standard" if index == 0 else "variant_" + str(index)
            },
            "traffic_allocation": f"{100 // len(campaign.ad_groups)}%",
            "budget_allocation": f"{100 // len(campaign.ad_groups)}%"
        }
    
    def _create_audience_variant(
        self,
        variant_id: str,
        variant_name: str,
        index: int,
        campaign: Campaign
    ) -> Dict[str, Any]:
        """Crea variante para testing de audiencias"""
        audience_segments = [
            "broad_audience",
            "interest_based",
            "behavior_based", 
            "demographic_focused",
            "lookalike_audience"
        ]
        
        segment = audience_segments[index % len(audience_segments)]
        
        return {
            "variant_id": variant_id,
            "variant_name": variant_name,
            "segment_type": segment,
            "description": f"Testing segmento: {segment.replace('_', ' ').title()}",
            "targeting_criteria": {
                "segment_type": segment,
                "audience_size": "medium",
                "precision_level": "medium"
            },
            "traffic_allocation": f"{100 // len(campaign.ad_groups)}%",
            "budget_allocation": f"{100 // len(campaign.ad_groups)}%"
        }
    
    def _create_bidding_variant(
        self,
        variant_id: str,
        variant_name: str,
        index: int,
        campaign: Campaign
    ) -> Dict[str, Any]:
        """Crea variante para testing de bidding"""
        bidding_strategies = [
            "manual_cpc",
            "target_cpa",
            "target_roas",
            "maximize_clicks",
            "maximize_conversions"
        ]
        
        strategy = bidding_strategies[index % len(bidding_strategies)]
        
        return {
            "variant_id": variant_id,
            "variant_name": variant_name,
            "bidding_strategy": strategy,
            "description": f"Testing estrategia: {strategy.replace('_', ' ').title()}",
            "bid_configuration": {
                "strategy": strategy,
                "bid_limit": "auto",
                "optimization_goal": "conversions"
            },
            "traffic_allocation": f"{100 // len(campaign.ad_groups)}%",
            "budget_allocation": f"{100 // len(campaign.ad_groups)}%"
        }
    
    def _create_generic_variant(
        self,
        variant_id: str,
        variant_name: str,
        index: int,
        campaign: Campaign
    ) -> Dict[str, Any]:
        """Crea variante genérica"""
        return {
            "variant_id": variant_id,
            "variant_name": variant_name,
            "description": f"Variante de control #{index + 1}",
            "configuration": "baseline",
            "traffic_allocation": f"{100 // len(campaign.ad_groups)}%",
            "budget_allocation": f"{100 // len(campaign.ad_groups)}%"
        }
    
    def _calculate_test_budget(
        self,
        campaign: Campaign,
        num_variants: int,
        config: Dict
    ) -> Dict[str, Any]:
        """Calcula presupuesto para el A/B test"""
        total_campaign_budget = campaign.total_budget
        
        # Usar 20-30% del presupuesto para testing
        test_budget_percentage = 0.25
        total_test_budget = total_campaign_budget * test_budget_percentage
        
        # Distribuir equitativamente entre variantes
        budget_per_variant = total_test_budget / num_variants
        daily_test_budget = total_test_budget / config["test_duration_days"]
        
        return {
            "total_test_budget": round(total_test_budget, 2),
            "budget_per_variant": round(budget_per_variant, 2),
            "daily_test_budget": round(daily_test_budget, 2),
            "budget_percentage": test_budget_percentage * 100,
            "daily_budget_per_variant": round(daily_test_budget / num_variants, 2)
        }
    
    def _generate_evaluation_criteria(self, config: Dict) -> Dict[str, Any]:
        """Genera criterios de evaluación para el test"""
        return {
            "primary_metrics": config["metrics"],
            "success_criteria": {
                "min_improvement": "10%",  # Mínimo 10% de mejora
                "statistical_significance": config["significance_level"],
                "min_sample_size": config["min_sample_size"]
            },
            "evaluation_method": "frequentist_statistical_test",
            "winner_selection": "highest_composite_score"
        }
    
    def _calculate_success_threshold(self, config: Dict) -> Dict[str, float]:
        """Calcula umbrales de éxito para el test"""
        return {
            "significance_level": config["significance_level"],
            "statistical_power": config["statistical_power"],
            "min_sample_size": config["min_sample_size"],
            "min_test_duration": config["test_duration_days"],
            "performance_improvement_threshold": 0.10  # 10% mejora mínima
        }


class AnchorClient:
    """Cliente para integración con datos de productos desde ANCHOR"""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    async def get_products_data(self, market: str, product_line: str) -> List[Dict]:
        """Obtiene datos de productos desde ANCHOR"""
        try:
            # Simular llamada a API de ANCHOR
            # En implementación real, usar: self.session.get(f"{self.base_url}/products")
            
            logger.info(f"Obteniendo datos de productos para {market} - {product_line}")
            
            # Datos simulados basados en la estructura esperada
            mock_products = await self._generate_mock_products(market, product_line)
            
            return mock_products
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de ANCHOR: {e}")
            # Retornar datos por defecto en caso de error
            return await self._generate_default_products()
    
    async def _generate_mock_products(self, market: str, product_line: str) -> List[Dict]:
        """Genera datos mock de productos para testing"""
        base_products = [
            {
                "id": "prod_001",
                "name": "Software CRM Pro",
                "category": "technology",
                "description": "Sistema completo de gestión de relaciones con clientes",
                "price": 99.99,
                "currency": "USD",
                "url": "https://example.com/crm-pro",
                "key_benefit": "Aumenta ventas en 30%",
                "main_benefit": "Gestión automatizada de leads",
                "specific_benefit": "Seguimiento de oportunidades",
                "features": ["Automatización de emails", "Pipeline visual", "Reportes avanzados"]
            },
            {
                "id": "prod_002", 
                "name": "Curso Marketing Digital",
                "category": "education",
                "description": "Aprende marketing digital desde cero hasta avanzado",
                "price": 299.00,
                "currency": "USD",
                "url": "https://example.com/marketing-course",
                "key_benefit": "Domina el marketing digital",
                "main_benefit": "Conocimientos prácticos",
                "specific_benefit": "Estrategias comprobadas",
                "features": ["50+ lecciones", "Casos reales", "Certificación"]
            },
            {
                "id": "prod_003",
                "name": "Consultoría Empresarial",
                "category": "business", 
                "description": "Consultoría especializada para crecimiento empresarial",
                "price": 500.00,
                "currency": "USD", 
                "url": "https://example.com/consulting",
                "key_benefit": "Crecimiento acelerado",
                "main_benefit": "Estrategias personalizadas",
                "specific_benefit": "ROI medible",
                "features": ["Análisis completo", "Plan de acción", "Seguimiento mensual"]
            },
            {
                "id": "prod_004",
                "name": "App Móvil Business",
                "category": "technology",
                "description": "Aplicación móvil para gestión empresarial",
                "price": 49.99,
                "currency": "USD",
                "url": "https://example.com/business-app",
                "key_benefit": "Gestión desde tu móvil",
                "main_benefit": "Acceso 24/7",
                "specific_benefit": "Notificaciones inteligentes",
                "features": ["Interfaz intuitiva", "Sincronización en la nube", "Análisis en tiempo real"]
            },
            {
                "id": "prod_005",
                "name": "Kit de Herramientas Productivity",
                "category": "business",
                "description": "Conjunto de herramientas para aumentar productividad",
                "price": 199.00,
                "currency": "USD", 
                "url": "https://example.com/productivity-kit",
                "key_benefit": "Duplica tu productividad",
                "main_benefit": "Automatización de tareas",
                "specific_benefit": "Integración con apps populares",
                "features": ["Plantillas personalizables", "Automatización", "Analytics"]
            }
        ]
        
        # Ajustar productos según mercado
        if market == "MX":
            for product in base_products:
                product["price"] = product["price"] * 20  # Conversión aproximada
                product["currency"] = "MXN"
        elif market == "ES":
            for product in base_products:
                product["price"] = product["price"] * 0.85  # Conversión aproximada
                product["currency"] = "EUR"
        
        return base_products
    
    async def _generate_default_products(self) -> List[Dict]:
        """Genera productos por defecto en caso de error de API"""
        return [
            {
                "id": "default_001",
                "name": "Producto MIDAS",
                "category": "general",
                "description": "Producto de alta calidad",
                "price": 99.99,
                "currency": "USD",
                "url": "https://example.com/default",
                "key_benefit": "Beneficio principal",
                "main_benefit": "Beneficio principal",
                "specific_benefit": "Beneficio específico"
            }
        ]
    
    async def sync_campaign_data(self, campaign: Campaign) -> Dict[str, Any]:
        """Sincroniza datos de campaña con ANCHOR"""
        try:
            # En implementación real: self.session.post(f"{self.base_url}/campaigns", data=campaign_data)
            
            sync_data = {
                "anchor.raw.feed": {
                    "market": campaign.config.market,
                    "product_line": campaign.config.product_line,
                    "campaign_title": f"Campaña {campaign.config.intent}",
                    "ad_copy": self._extract_ad_copy(campaign),
                    "assets_uri": f"gs://midas/ads/{campaign.campaign_id}/",
                    "trace_id": campaign.trace_id,
                    "sync_timestamp": datetime.datetime.utcnow().isoformat()
                },
                "anchor.raw.ads.daily": {
                    "market": campaign.config.market,
                    "product_line": campaign.config.product_line,
                    "platform": "google_ads",  # Ejemplo
                    "spend_total": 0.0,  # Se actualizará con datos reales
                    "impressions": 0,
                    "clicks": 0,
                    "leads_day": 0,
                    "trace_id": campaign.trace_id,
                    "status": "active"
                }
            }
            
            logger.info(f"Datos de campaña {campaign.campaign_id} sincronizados con ANCHOR")
            return sync_data
            
        except Exception as e:
            logger.error(f"Error sincronizando con ANCHOR: {e}")
            raise
    
    def _extract_ad_copy(self, campaign: Campaign) -> str:
        """Extrae ad copy de la campaña para ANCHOR"""
        all_creatives = []
        for ad_group in campaign.ad_groups:
            for creative in ad_group.creatives:
                all_creatives.append(f"Título: {creative.title}")
                all_creatives.append(f"Descripción: {creative.description}")
        
        return " | ".join(all_creatives[:3])  # Primeros 3 creativos


# Sistema principal de orquestación
class MidasCampaignGenerator:
    """Orquestador principal del sistema de generación de campañas"""
    
    def __init__(self, anchor_api_url: str = "https://api.anchor.com", api_key: str = None):
        self.template_engine = CampaignTemplate(anchor_api_url, api_key)
        self.targeting_engine = TargetingEngine()
        self.copy_generator = AdCopyGenerator()
        self.budget_calculator = BudgetCalculator()
        self.scheduler = CampaignScheduler()
        self.ab_test_manager = ABTestManager()
        self.anchor_client = AnchorClient(anchor_api_url, api_key)
        
        logger.info("Midas Campaign Generator inicializado")
    
    async def generate_complete_campaign(
        self,
        campaign_config: CampaignConfig,
        template_name: str = "product_launch",
        enable_ab_testing: bool = True
    ) -> Dict[str, Any]:
        """Genera una campaña completa con todos los componentes"""
        
        logger.info(f"Iniciando generación de campaña completa: {campaign_config.cmd_id}")
        
        try:
            # 1. Generar campaña desde template
            campaign = await self.template_engine.create_campaign_from_template(
                template_name, 
                campaign_config
            )
            
            # 2. Generar targeting automático
            product_data = await self.anchor_client.get_products_data(
                campaign_config.market,
                campaign_config.product_line
            )
            
            targeting = await self.targeting_engine.generate_targeting_criteria(
                campaign_config,
                product_data[0] if product_data else {}
            )
            campaign.config.targeting = targeting
            
            # 3. Calcular presupuesto automático
            target_audience_size = campaign.estimated_reach
            budget_plan = self.budget_calculator.calculate_budget_allocation(
                campaign_config,
                target_audience_size
            )
            
            # 4. Generar programación automática
            schedule_plan = self.scheduler.generate_campaign_schedule(
                campaign_config,
                target_audience_type="b2c"  # Podría inferirse del producto
            )
            
            # 5. Crear plan de A/B testing si está habilitado
            ab_test_plan = None
            if enable_ab_testing and campaign.config.ab_testing:
                ab_test_plan = self.ab_test_manager.create_ab_test_plan(
                    campaign,
                    test_type="creative_testing",
                    num_variants=3
                )
            
            # 6. Sincronizar con ANCHOR
            anchor_sync_data = await self.anchor_client.sync_campaign_data(campaign)
            
            # 7. Compilar campaña completa
            complete_campaign = {
                "campaign": asdict(campaign),
                "targeting": asdict(targeting),
                "budget_plan": budget_plan,
                "schedule_plan": schedule_plan,
                "ab_test_plan": ab_test_plan,
                "anchor_sync": anchor_sync_data,
                "generation_metadata": {
                    "generated_at": datetime.datetime.utcnow().isoformat(),
                    "template_used": template_name,
                    "midas_version": "1.0.0",
                    "total_estimated_reach": campaign.estimated_reach,
                    "total_estimated_clicks": campaign.estimated_clicks,
                    "total_estimated_conversions": campaign.estimated_conversions,
                    "ab_testing_enabled": enable_ab_testing
                }
            }
            
            logger.info(f"Campaña completa generada exitosamente: {campaign.campaign_id}")
            return complete_campaign
            
        except Exception as e:
            logger.error(f"Error generando campaña completa: {e}")
            raise
    
    async def batch_generate_campaigns(
        self,
        campaign_configs: List[CampaignConfig],
        template_name: str = "product_launch"
    ) -> List[Dict[str, Any]]:
        """Genera múltiples campañas en lote"""
        
        logger.info(f"Iniciando generación en lote de {len(campaign_configs)} campañas")
        
        results = []
        for i, config in enumerate(campaign_configs):
            try:
                campaign = await self.generate_complete_campaign(config, template_name)
                results.append(campaign)
                logger.info(f"Campaña {i+1}/{len(campaign_configs)} completada: {config.cmd_id}")
            except Exception as e:
                logger.error(f"Error en campaña {i+1}: {e}")
                # Continuar con la siguiente campaña
                continue
        
        logger.info(f"Generación en lote completada: {len(results)}/{len(campaign_configs)} exitosas")
        return results
    
    def export_campaign_data(self, campaign_data: Dict[str, Any], format: str = "json") -> str:
        """Exporta datos de campaña en diferentes formatos"""
        
        if format.lower() == "json":
            return json.dumps(campaign_data, indent=2, ensure_ascii=False)
        
        elif format.lower() == "csv":
            # Convertir a CSV para importación en Google Ads/Meta Ads
            return self._export_to_csv(campaign_data)
        
        elif format.lower() == "google_ads":
            # Formato específico para Google Ads
            return self._export_to_google_ads_format(campaign_data)
        
        else:
            raise ValueError(f"Formato de exportación no soportado: {format}")
    
    def _export_to_csv(self, campaign_data: Dict[str, Any]) -> str:
        """Convierte datos de campaña a formato CSV"""
        csv_lines = ["Campaign,Ad Group,Keyword,Headline,Description,Final URL,Daily Budget"]
        
        campaign = campaign_data["campaign"]
        for ad_group in campaign["ad_groups"]:
            for creative in ad_group["creatives"]:
                line = f'"{campaign["config"]["intent"]}",'
                line += f'"{ad_group["name"]}",'
                line += f'"{",".join(ad_group["keywords"][:3])}",'  # Primeras 3 keywords
                line += f'"{creative["headline"]}",'
                line += f'"{creative["description"]}",'
                line += f'"{creative["final_url"]}",'
                line += f'"{ad_group["daily_budget"]}"'
                csv_lines.append(line)
        
        return "\n".join(csv_lines)
    
    def _export_to_google_ads_format(self, campaign_data: Dict[str, Any]) -> str:
        """Convierte datos al formato de importación de Google Ads"""
        # Estructura simplificada para Google Ads Editor
        export_data = {
            "Campaigns": [{
                "Name": campaign_data["campaign"]["config"]["intent"],
                "Budget": f"${campaign_data["campaign"]["total_budget"]}",
                "Campaign Type": "Search",
                "Status": "Enabled"
            }],
            "Ad Groups": [],
            "Ads": [],
            "Keywords": []
        }
        
        campaign = campaign_data["campaign"]
        for ad_group in campaign["ad_groups"]:
            # Ad Group
            export_data["Ad Groups"].append({
                "Name": ad_group["name"],
                "Campaign": campaign["config"]["intent"],
                "Status": "Enabled"
            })
            
            # Keywords
            for keyword in ad_group["keywords"][:10]:  # Máximo 10 keywords
                export_data["Keywords"].append({
                    "Keyword": keyword,
                    "Match Type": "Broad",
                    "Ad Group": ad_group["name"],
                    "Campaign": campaign["config"]["intent"]
                })
            
            # Ads
            for creative in ad_group["creatives"]:
                export_data["Ads"].append({
                    "Headline 1": creative["headline"][:30],
                    "Headline 2": creative["headline"][30:60] if len(creative["headline"]) > 30 else "",
                    "Headline 3": creative["headline"][60:90] if len(creative["headline"]) > 60 else "",
                    "Description": creative["description"][:90],
                    "Final URL": creative["final_url"],
                    "Ad Group": ad_group["name"],
                    "Campaign": campaign["config"]["intent"]
                })
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)


# Función principal de ejemplo
async def main():
    """Función principal de ejemplo y testing"""
    
    # Configurar ejemplo de campaña
    campaign_config = CampaignConfig(
        cmd_id="cmd-2025-11-06-001",
        intent="launch_campaign",
        market="US",
        product_line="technology",
        budget=1000.0,
        duration_days=7,
        objective=CampaignObjective.CONVERSIONS,
        platform=[CampaignPlatform.GOOGLE_ADS, CampaignPlatform.META_ADS],
        ad_formats=[AdFormat.TEXT_AD, AdFormat.IMAGE_AD],
        start_date=datetime.datetime.now(),
        ab_testing=True
    )
    
    # Inicializar generador
    generator = MidasCampaignGenerator()
    
    try:
        # Generar campaña completa
        campaign_data = await generator.generate_complete_campaign(
            campaign_config,
            template_name="product_launch",
            enable_ab_testing=True
        )
        
        # Mostrar resultados
        print("=== CAMPAÑA GENERADA EXITOSAMENTE ===")
        print(f"ID de Campaña: {campaign_data['campaign']['campaign_id']}")
        print(f"Objetivo: {campaign_data['campaign']['config']['objective']}")
        print(f"Presupuesto Total: ${campaign_data['campaign']['total_budget']}")
        print(f"Alcance Estimado: {campaign_data['campaign']['estimated_reach']:,}")
        print(f"Clicks Estimados: {campaign_data['campaign']['estimated_clicks']:,}")
        print(f"Conversiones Estimadas: {campaign_data['campaign']['estimated_conversions']:,}")
        print(f"Grupos de Anuncios: {len(campaign_data['campaign']['ad_groups'])}")
        print(f"A/B Testing: {'Habilitado' if campaign_data['generation_metadata']['ab_testing_enabled'] else 'Deshabilitado'}")
        
        # Exportar datos
        json_export = generator.export_campaign_data(campaign_data, "json")
        csv_export = generator.export_campaign_data(campaign_data, "csv")
        
        print("\n=== EXPORTACIÓN COMPLETADA ===")
        print(f"Datos JSON: {len(json_export)} caracteres")
        print(f"Datos CSV: {len(csv_export)} caracteres")
        
        # Guardar archivos
        with open(f"campaign_{campaign_data['campaign']['campaign_id']}.json", "w") as f:
            f.write(json_export)
        
        with open(f"campaign_{campaign_data['campaign']['campaign_id']}.csv", "w") as f:
            f.write(csv_export)
        
        print("Archivos guardados exitosamente")
        
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Error en main: {e}")


if __name__ == "__main__":
    # Ejecutar ejemplo
    asyncio.run(main())
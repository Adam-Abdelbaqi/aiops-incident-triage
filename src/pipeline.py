from src.features.feature_engineering import TabularExtractor
from src.features.feature_extraction import TransformerExtractor
from src.features.dim_reducer import EmbeddingReduction
from src.models.classifier import RandomForestModel
from sklearn.pipeline import Pipeline , FeatureUnion



def build_pipeline():
    """Combine text features, tabular features, and the classifier."""

    text_pipeline = Pipeline(
        steps=[
            (
                "text_extractor",
                TransformerExtractor(
                    text_columns=("subject", "description"),
                    model_name="all-MiniLM-L6-v2",
                    batch_size=32,
                ),
            ),
            (
                "embedding_reduction",
                EmbeddingReduction(
                    n_components=30,
                    random_state=42,
                ),
            ),
        ])

    # The text columns are removed here because they are handled by the text branch.
    # Keep product_service out of X as decided during feature selection.
    tabular_pipeline = Pipeline(
        steps=[
            (
                "tabular_extractor",
                TabularExtractor(
                    cols_to_drop=[
                        "status",
                        "category",
                        "satisfaction_rating",
                        "resolved_date",
                        "ticket_id",
                        "resolution",
                        "subject",
                        "description",
                    ]
                ),
            ),
        ]
    )

    combined_features = FeatureUnion(
        transformer_list=[
            ("text_features", text_pipeline),
            ("tabular_features", tabular_pipeline),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("features", combined_features),
            (
                "classifier",
                RandomForestModel(
                    n_estimators=200,
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline

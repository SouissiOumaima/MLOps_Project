"""
Point d'entrée principal du pipeline MLOps Titanic.
"""

import argparse
from src.model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    compare_models,
    logger,
)


def main() -> None:
    """Point d'entrée principal du pipeline avec gestion des arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline MLOps Titanic - Exécution modulaire des étapes"
    )

    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Exécute uniquement l'étape de préparation des données",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Exécute l'entraînement du modèle (inclut prepare_data)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Exécute l'évaluation du modèle",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Exécute tout le pipeline (préparation, entraînement, évaluation, sauvegarde)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare LogisticRegression, RandomForest et SVM — sauvegarde le meilleur",
    )

    args = parser.parse_args()

    logger.info("Démarrage du pipeline Titanic MLOps")

    # ── Mode par défaut : --all si aucun flag n'est spécifié ─────────────────
    if args.all or not (args.prepare or args.train or args.evaluate or args.compare):
        X_train, X_test, y_train, y_test = prepare_data()
        model = train_model(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test)
        save_model(model)
        return

    # ── Exécution sélective des étapes ───────────────────────────────────────
    if args.prepare:
        prepare_data()
        logger.info("Étape préparation des données terminée avec succès.")

    if args.train:
        X_train, X_test, y_train, y_test = prepare_data()
        model = train_model(X_train, y_train, X_test, y_test)
        save_model(model)
        logger.info("Étape entraînement du modèle terminée avec succès.")

    if args.evaluate:
        X_train, X_test, y_train, y_test = prepare_data()
        model = train_model(X_train, y_train, X_test, y_test)
        evaluate_model(model, X_test, y_test)

    if args.compare:
        logger.info("Lancement de la comparaison des modèles...")
        X_train, X_test, y_train, y_test = prepare_data()
        results = compare_models(X_train, y_train, X_test, y_test)

        print("\n╔══════════════════════════════════════════╗")
        print("║      Résultats de la comparaison         ║")
        print("╠══════════════════════════════════════════╣")
        for rank, (name, info) in enumerate(results.items(), 1):
            medal = ["🥇", "🥈", "🥉"][rank - 1] if rank <= 3 else "  "
            print(f"║  {medal}  {rank}. {name:<22} {info['accuracy']:.4f}  ║")
        print("╚══════════════════════════════════════════╝")

        best_name = list(results.keys())[0]
        best_acc  = results[best_name]["accuracy"]
        print(f"\n✔  Meilleur modèle : {best_name} (accuracy = {best_acc:.4f})")
        print("✔  Modèle sauvegardé dans models/titanic_model.pkl")


if __name__ == "__main__":
    main()
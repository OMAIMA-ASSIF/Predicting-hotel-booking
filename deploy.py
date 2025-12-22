from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os

# Définir les fonctions nécessaires
def sigmoid(x):
    """Fonction sigmoid pour la régression logistique"""
    x = np.clip(x, -20, 20)  # Éviter l'overflow
    return 1 / (1 + np.exp(-x))

app = Flask(__name__)

# Charger le modèle
def load_model():
    try:
        with open('hotel_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            
        # Vérifier si le modèle contient des fonctions non sérialisables
        if 'sigmoid' in model_data:
            print("⚠️  Modèle contient des fonctions, utilisation de nos fonctions locales")
            # Utiliser notre fonction sigmoid locale
            model_data['sigmoid'] = sigmoid
            
        return model_data
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        # Essayer de charger un modèle fixé
        try:
            with open('hotel_model_fixed.pkl', 'rb') as f:
                print("✅ Modèle fixé chargé avec succès!")
                return pickle.load(f)
        except:
            print("❌ Aucun modèle valide trouvé")
            return None

model_data = load_model()
def normalize_model_weights(model_data, scale=0.1):
    """Normalise les poids du modèle pour éviter la saturation"""
    model_data['W'] = model_data['W'] * scale
    model_data['b'] = model_data['b'] * scale
    return model_data


if model_data:
    print("✅ Modèle chargé avec succès!")
    model_data = normalize_model_weights(model_data, scale=0.05)  # Ajustez ce facteur

    print("🔍 Contenu du modèle:")
    for key in model_data.keys():
        print(f"   {key}: {type(model_data[key])}")
else:
    print("❌ Modèle non chargé")

# Les mappings (gardez les vôtres)
meal_plan_mapping = {
    'Meal Plan 1': 0,
    'Meal Plan 2': 1, 
    'Meal Plan 3': 2,
    'Not Selected': 3
}

room_type_mapping = {
    'Room_Type 1': 0,
    'Room_Type 2': 1,
    'Room_Type 3': 2,
    'Room_Type 4': 3,
    'Room_Type 5': 4,
    'Room_Type 6': 5,
    'Room_Type 7': 6
}

market_segment_mapping = {
    'Aviation': 0,
    'Complementary': 1,
    'Corporate': 2,
    'Offline': 3,
    'Online': 4
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model_data is None:
            return render_template('index.html', error="Modèle non chargé")
        
        # Récupérer les données du formulaire (votre code existant)
        no_of_adults = int(request.form['no_of_adults'])
        no_of_children = int(request.form['no_of_children'])
        no_of_weekend_nights = int(request.form['no_of_weekend_nights'])
        no_of_week_nights = int(request.form['no_of_week_nights'])
        
        # Variables catégorielles
        type_of_meal_plan = meal_plan_mapping[request.form['type_of_meal_plan']]
        room_type_reserved = room_type_mapping[request.form['room_type_reserved']]
        market_segment_type = market_segment_mapping[request.form['market_segment_type']]
        
        required_car_parking_space = int(request.form['required_car_parking_space'])
        lead_time = int(request.form['lead_time'])
        arrival_month = int(request.form['arrival_month'])
        arrival_date = int(request.form['arrival_date'])
        repeated_guest = int(request.form['repeated_guest'])
        no_of_previous_cancellations = int(request.form['no_of_previous_cancellations'])
        no_of_previous_bookings_not_canceled = int(request.form['no_of_previous_bookings_not_canceled'])
        avg_price_per_room = float(request.form['avg_price_per_room'])
        no_of_special_requests = int(request.form['no_of_special_requests'])
        
        # Créer le DataFrame
        input_data = {
            'no_of_adults': no_of_adults,
            'no_of_children': no_of_children,
            'no_of_weekend_nights': no_of_weekend_nights,
            'no_of_week_nights': no_of_week_nights,
            'type_of_meal_plan': type_of_meal_plan,
            'required_car_parking_space': required_car_parking_space,
            'room_type_reserved': room_type_reserved,
            'lead_time': lead_time,
            'arrival_month': arrival_month,
            'arrival_date': arrival_date,
            'market_segment_type': market_segment_type,
            'repeated_guest': repeated_guest,
            'no_of_previous_cancellations': no_of_previous_cancellations,
            'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
            'avg_price_per_room': avg_price_per_room,
            'no_of_special_requests': no_of_special_requests
        }
        
        # Convertir en DataFrame
        X_input = pd.DataFrame([input_data])
        X_input = X_input[model_data['feature_names']]
        
        print(f"🔍 Données avant normalisation:")
        print(f"   - Lead time: {lead_time}")
        print(f"   - Prix moyen: {avg_price_per_room}")
        
        # Normalisation
        X_norm = (X_input - model_data['mean']) / model_data['std']

        # Transformation polynomiale
        X_poly = model_data['poly'].transform(X_norm)

        # Prédiction
        Z = np.dot(X_poly, model_data['W']) + model_data['b']
        
        #Clipping de Z pour éviter la saturation de la sigmoïde
        Z_clipped = np.clip(Z, -10, 10)  # Limite Z entre -10 et 10
        
        probability = 1 / (1 + np.exp(-Z_clipped))  # Utiliser notre sigmoid
        
        
        
        def rescale_probability(prob, target_min=0.2, target_max=0.8):
            """Rééchelonne les probabilités pour qu'elles soient entre target_min et target_max"""
            if prob > 0.5:
                # Prob > 0.5 : échelle 0.5-1 → target_min-1
                return target_min + (prob - 0.5) * (1 - target_min) * 2
            else:
                # Prob < 0.5 : échelle 0-0.5 → 0-target_max
                return prob * target_max * 2
        #Arrondir pour éviter les 100% exacts
        prob_confirm  = float(probability[0][0])
        prob_confirm  = np.clip(prob_confirm , 0.01, 0.99)  # Éviter 0% et 100%
        prob_confirm_rescaled = rescale_probability(prob_confirm, 0.3, 0.7)
        prob_confirm = prob_confirm_rescaled
        
        
        prob_cancel = 1 - prob_confirm 
        # Décision
        prediction = int(prob_confirm  >= 0.5)
        
        # Interprétation
        if prediction == 1:
            result_message = "La réservation sera CONFIRMÉE"
            display_probability = prob_confirm * 100  # Probabilité de confirmation
            confidence = prob_confirm * 100  # Pour la barre de progression


        else:
            result_message = "La réservation sera ANNULÉE"
            display_probability = prob_cancel * 100  # Probabilité d'annulation
            confidence = prob_cancel * 100  # Pour la barre de progression

        
        print(f"🔍 Debug prédiction:")
        print(f"   - Probabilité confirmation: {prob_confirm:.4f} ({prob_confirm*100:.2f}%)")
        print(f"   - Probabilité annulation: {prob_cancel:.4f} ({prob_cancel*100:.2f}%)")
        print(f"   - Prédiction: {'Confirmée' if prediction == 1 else 'Annulée'}")
        print(f"   - Affichage: {display_probability:.2f}%")
        
        
        return render_template('index.html', 
                             result=prediction,
                             result_message=result_message,
                             confidence=round(confidence, 2),
                             probability=round(display_probability, 2),
                             form_data={
                                 'lead_time': lead_time,
                                 'avg_price_per_room': avg_price_per_room,
                                 'repeated_guest': repeated_guest,
                                 'no_of_previous_cancellations': no_of_previous_cancellations,
                                 'no_of_special_requests': no_of_special_requests,
                                 'room_type_reserved': request.form['room_type_reserved'],
                                 'type_of_meal_plan': request.form['type_of_meal_plan'],
                                 'required_car_parking_space': required_car_parking_space,
                                 'market_segment_type': request.form['market_segment_type'],
                                 'arrival_month': arrival_month
                             },
                             has_form_data=True)
                             
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"Erreur: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
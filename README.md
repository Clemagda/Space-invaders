# AI Learn to play SpaceInvaders thank to Reinforcment Learning




Le modèle a été entrainé en utilisant la fonction de perte Hubert ``keras.losses.Huber()`` . Ce choix est justifié par la robustesse  aux valeurs aberrantes de la fonction. Dans une démarche de reinforcement learning (RL) les Q-values peuvent subir de brusques variations ou prendre des valeurs extrêmes suite à  des récompenses imprévues ou des transitions de valeurs rares. COntraitement aux fonction L1 et L2, La perte de Huber est moins sensible à ces valeurs extrêmes conduisant ainsi à un entraînement plus stable. A partir d'un seuil défini, la perte de Huber assure un arbitrage entre les pertes L1 et L2. De cette sorte, la fonction utilise L2 pour les petites erreurs (ce qui est bon pour l'apprentissage rapide et stable) et L1 pour les grandes erreurs (ce qui réduit l'impact des valeurs aberrantes).

Projet basé sur l'utilisation des algorithmes Deep Q-Learning développés dans l'article [<i>Playing Atari with Deep Reinforcement Learning</i>](https://doi.org/10.48550/arXiv.1312.5602)

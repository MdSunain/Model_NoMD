FEDERATED LEARNING
Prerequisite: to download dataset, you can download using this link to my google drive 
Folder name: federated_data (Please Do Not Change the Folder Name)


https://drive.google.com/drive/folders/1IaNSS8D4ci1cO2vVzT9GRNvcDP41IgiI?usp=sharing

install requirements -> 
# python -m venv .venv
# .venv\Scripts\activate (in Command prompt cmd)

# pip install -r requirement.txt

step 1: Defining a Model in <model.js>

Step 2: Training a Client, How to use this Model in <client_train.py>

Step 3: Defining a Server to ~Average~ the Client weights(OUTPUT) in <server.py>

Step 4: running federated architecture for 'r' rounds in <federated_final.py>

🎉Congratulation.. we got a final_global_model which has averaged clients_weights for r interations

Sample OUTPUT: 

i.  for client1: we see Loss Decreasing gradually in each round
    Round 1 --> Epoch1: Loss = 18.5840
                , Epoch5: Loss = 14.1656 | 
    Round 2 --> Epoch1: Loss = 14.8394
                , Epoch5: Loss = 7.6383 | 
    Round 3 --> Epoch1: Loss = 9.8005
                , Epoch5: Loss = 4.4412 |
    Improvement from ~ 18 --> ~ 4 //  <a. Round Wise Loss Improvement>

ii. finally
    client1 --> Loss = ~ 4.5   (before: ~ 18.5)
    client2 --> Loss = ~ 5.5   (before: ~ 18.0)
    client3 --> Loss = ~ 5.0   (before: ~ 18.1)

* They are close
* They improve at similar rates
* Bias is reducing
<b. This means the global model is stabilizing across non-IID data.>




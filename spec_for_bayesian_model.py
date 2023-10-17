import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import cProfile

# Define the logistic regression model
def logistic_regression(data, labels):
    num_samples, num_features = data.shape

    # Define the weights prior
    weight_prior = dist.Normal(torch.zeros(num_features), torch.ones(num_features))

    # Sample the weights from the prior
    with pyro.plate("weights_plate", num_features):
        weights = pyro.sample("weights", weight_prior)

    # Compute the logits
    logits = torch.matmul(data, weights)

    # Define the likelihood
    likelihood = dist.Bernoulli(logits=logits) # likelihood of heads

    # Sample the labels from the likelihood
    with pyro.plate("data_plate", num_samples):
        pyro.sample("obs", likelihood, obs=labels)

# Define the guide
def guide(data, labels):
    num_features = data.shape[1]

    # Define the weights posterior
    weight_posterior_loc = pyro.param("weight_posterior_loc", torch.randn(num_features))
    weight_posterior_scale = pyro.param("weight_posterior_scale", torch.ones(num_features),
                                        constraint=dist.constraints.positive)
    weight_posterior = dist.Normal(weight_posterior_loc, weight_posterior_scale)

    # Sample the weights from the posterior
    with pyro.plate("weights_plate", num_features):
        pyro.sample("weights", weight_posterior)

# Train the model using Stochastic Variational Inference (SVI)
def train(data, labels, num_epochs=1000):
    # Clear any existing parameters
    pyro.clear_param_store()

    # Define the optimizer
    optimizer = Adam({"lr": 0.01})

    # Define the SVI object
    svi = SVI(logistic_regression, guide, optimizer, loss=Trace_ELBO())

    # Train the model for a specified number of epochs
    for epoch in range(num_epochs):
        loss = svi.step(data, labels)
        if epoch % 100 == 0:
            print("Epoch ", epoch, " Loss: ", loss)

    # Return the learned weights
    weight_posterior = dist.Normal(pyro.param("weight_posterior_loc"),
                                   pyro.param("weight_posterior_scale"))
    print(type(weight_posterior),"is type of weight posterior")
    print(weight_posterior.entropy(), "is the entropy of the posterior")
    return weight_posterior.sample()

# Generate some dummy data
data = torch.randn(100, 10)
true_weights = torch.randn(10)
logits = torch.matmul(data, true_weights)
labels = dist.Bernoulli(logits=logits).sample()

# Train the model
learned_weights = cProfile.run(train(data, labels))

print("True weights: ", true_weights)
print("Learned weights: ", learned_weights)

# Define a function to make predictions using the trained model
def predict(data, learned_weights):
    num_samples, num_features = data.shape

    # Define the weights posterior
    weight_posterior = dist.Normal(pyro.param("weight_posterior_loc"),
                                   pyro.param("weight_posterior_scale"))

    # Compute the logits using the learned weights
    logits = torch.matmul(data, learned_weights)

    # Compute the predicted probabilities
    predicted_probs = torch.sigmoid(logits)

    return predicted_probs

# Generate some new data to make predictions on
new_data = torch.randn(10, 10)

# Make predictions using the trained model
predicted_probs = predict(new_data, learned_weights)

print("Predicted probabilities: ", predicted_probs)

import torch
import torch.nn.functional as F

# Computes the KL divergence between two diagonal Gaussian distributions:
# one representing the posterior and one the prior.
def kl_loss(z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar):
    # COMPUTE KL DIV
    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 0.5 * (z_prior_logvar - z_post_logvar +
                   ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2)) / z_prior_var) - 1)
    return kld_z

# Computes an eigenvalue regularization loss on the prior transition matrix.
# Encourages the largest eigenvalues to be close to 1, depending on the selected mode:
# - mode '2': top 2 eigenvalues
# - mode '3': top 3 eigenvalues
# - mode '4': all eigenvalues
def eig_loss(Ct_prior, mode):
    # COMPUTE EIG PENALTY:
    eigs = torch.abs(torch.linalg.eigvals(Ct_prior))
    if mode == '2':
        eigs_latent = torch.argsort(eigs)[-2:]
        one_valued_eig = torch.ones_like(eigs[eigs_latent], device=eigs.device, dtype=torch.float32)
        eig_loss_prior = F.mse_loss(eigs[eigs_latent], one_valued_eig)
    if mode == '3':
        eigs_latent = torch.argsort(eigs)[-3:]
        one_valued_eig = torch.ones_like(eigs[eigs_latent], device=eigs.device, dtype=torch.float32)
        eig_loss_prior = F.mse_loss(eigs[eigs_latent], one_valued_eig)
    if mode == '4':
        one_valued_eig = torch.ones_like(eigs, device=eigs.device, dtype=torch.float32)
        eig_loss_prior = F.mse_loss(eigs, one_valued_eig)

    return eig_loss_prior

# Extracts and compares the eigenvalues of the posterior and prior transition matrices.
# Splits them into two groups:
# - a portion expected to be close to 1 (top 50%)
# - the rest expected to be less than 1
# Returns both groups and a tensor of ones for loss computation.
def eigen_constraints(Ct_post, Ct_prior):
    eig_post = torch.abs(torch.linalg.eigvals(Ct_post))
    eig_prior = torch.abs(torch.linalg.eigvals(Ct_prior))
    eig_post_sorted = torch.argsort(eig_post, descending=True)
    eig_prior_sorted = torch.argsort(eig_prior, descending=True)
    eig_norm_one = .5
    eigs_to_one = int(eig_norm_one * len(eig_post_sorted))
    eigs_to_one_post, eigs_less_than_one_post = eig_post[eig_post_sorted[:eigs_to_one]], \
        eig_post[eig_post_sorted[eigs_to_one:]]
    eigs_to_one_prior, eigs_less_than_one_prior = eig_prior[eig_prior_sorted[:eigs_to_one]], \
        eig_prior[eig_prior_sorted[eigs_to_one:]]
    one_valued_eig = torch.ones_like(eigs_to_one_post)
    return eigs_less_than_one_post, eigs_less_than_one_prior, eigs_to_one_post, eigs_to_one_prior, one_valued_eig


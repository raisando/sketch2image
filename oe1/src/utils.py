def sample_batch(model, scheduler, cond, steps, device):
    model.eval()
    with torch.no_grad():
        # Configurar timesteps de muestreo (p.ej., 50)
        scheduler.set_timesteps(steps, device=device)
        b, c, h, w = cond.shape
        x = torch.randn((b, 3, h, w), device=device)
        for t in scheduler.timesteps:
            # Predicción de ruido ε
            eps = model(x, t, cond)
            # Un paso de denoise
            out = scheduler.step(model_output=eps, timestep=t, sample=x)
            x = out.prev_sample
        return x

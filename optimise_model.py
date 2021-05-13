from utils import copy_gradients


def optimise_model(shared_model, local_model, loss, optimiser, args, lock):
    # Compute gradients
    loss.backward()

    # The critical section begins
    lock.acquire()
    copy_gradients(shared_model, local_model)
    optimiser.step()
    lock.release()
    # The critical section ends

    local_model.zero_grad()

    return loss.item()

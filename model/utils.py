import os

from matplotlib import pyplot as plt


def save_fig_loss(train_loss, val_loss, dir):
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Błąd treningowy i walidacyjny')
    plt.legend()
    plt.grid(True)
    plt.show()
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, "Loss"))


def save_fig_energy(y_test, y_pred, dir):
    plt.plot(y_test, label='Real consumption')
    plt.plot(y_pred, label='Predicted')
    plt.xlabel('Hour')
    plt.ylabel('Consumption')
    plt.legend()
    plt.show()
    plt.grid(True)
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, "Energy"))


def save_fig(train_loss, val_loss, y_test, y_pred, test_loss, dir):
    epochs = range(1, len(train_loss) + 1)

    # Ustawienie subplotów
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)

    # Pierwszy wykres dla straty treningowej i walidacyjnej
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoka')
    plt.ylabel('Loss')
    plt.title('Błąd treningowy i walidacyjny')
    plt.legend()
    plt.grid(True)

    # Drugi wykres dla danych energetycznych
    plt.subplot(1, 2, 2)
    plt.plot(y_test, label='Realne')
    plt.plot(y_pred, label='Predykcja')
    plt.xlabel('Godzina')
    plt.ylabel('Zużycie energii [MW]')
    plt.title(f'Zużycie energii, Loss: {test_loss:.5f}')
    plt.legend()
    plt.grid(True)

    # Wyświetlenie i zapisanie wykresów
    plt.tight_layout()  # Poprawa układu wykresów
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, "Loss"))
    plt.show()

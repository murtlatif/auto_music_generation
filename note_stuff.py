from data.notes.simple_notes_dataset import SimpleNotes
from constants import OLD_MCDONALD_NOTES


def note_main():
    notes_dataset = SimpleNotes(OLD_MCDONALD_NOTES)
    return notes_dataset


if __name__ == '__main__':
    note_main()

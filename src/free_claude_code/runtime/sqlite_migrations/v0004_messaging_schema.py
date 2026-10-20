"""Relational messaging history and the retired JSON import receipt."""

import sqlite3


def upgrade(connection: sqlite3.Connection) -> None:
    connection.execute("""
        CREATE TABLE messaging_trees (
            platform TEXT NOT NULL, chat_id TEXT NOT NULL, root_id TEXT NOT NULL,
            PRIMARY KEY (platform, chat_id, root_id),
            FOREIGN KEY (platform, chat_id, root_id)
                REFERENCES messaging_nodes(platform, chat_id, node_id)
                DEFERRABLE INITIALLY DEFERRED
        )
    """)
    connection.execute("""
        CREATE TABLE messaging_nodes (
            platform TEXT NOT NULL, chat_id TEXT NOT NULL, node_id TEXT NOT NULL,
            root_id TEXT NOT NULL, parent_id TEXT, parent_reference_id TEXT,
            status_message_id TEXT, session_id TEXT,
            state TEXT NOT NULL CHECK (state IN ('pending','in_progress','completed','error')),
            PRIMARY KEY (platform, chat_id, node_id),
            UNIQUE (platform, chat_id, root_id, node_id),
            CHECK (length(node_id) > 0),
            CHECK (status_message_id IS NULL OR (length(status_message_id) > 0 AND status_message_id <> node_id)),
            CHECK (state NOT IN ('pending','in_progress') OR status_message_id IS NOT NULL),
            CHECK ((node_id = root_id AND parent_id IS NULL AND parent_reference_id IS NULL)
                OR (node_id <> root_id AND parent_id IS NOT NULL AND parent_reference_id IS NOT NULL AND parent_id <> node_id)),
            FOREIGN KEY (platform, chat_id, root_id)
                REFERENCES messaging_trees(platform, chat_id, root_id) ON DELETE CASCADE,
            FOREIGN KEY (platform, chat_id, root_id, parent_id)
                REFERENCES messaging_nodes(platform, chat_id, root_id, node_id)
                DEFERRABLE INITIALLY DEFERRED,
            FOREIGN KEY (platform, chat_id, parent_reference_id, parent_id)
                REFERENCES messaging_references(platform, chat_id, reference_id, node_id)
                DEFERRABLE INITIALLY DEFERRED
        )
    """)
    connection.execute("""
        CREATE TABLE messaging_references (
            platform TEXT NOT NULL, chat_id TEXT NOT NULL, reference_id TEXT NOT NULL,
            node_id TEXT NOT NULL, kind TEXT NOT NULL CHECK (kind IN ('prompt','status')),
            PRIMARY KEY (platform, chat_id, reference_id),
            UNIQUE (platform, chat_id, node_id, kind),
            UNIQUE (platform, chat_id, reference_id, node_id),
            FOREIGN KEY (platform, chat_id, node_id)
                REFERENCES messaging_nodes(platform, chat_id, node_id) ON DELETE CASCADE
        )
    """)
    connection.execute("""
        CREATE TRIGGER messaging_node_parent BEFORE INSERT ON messaging_nodes
        WHEN NEW.parent_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM messaging_nodes WHERE platform = NEW.platform AND chat_id = NEW.chat_id
                AND root_id = NEW.root_id AND node_id = NEW.parent_id)
        BEGIN SELECT RAISE(ABORT, 'Messaging parent must exist before its child'); END
    """)
    connection.execute("""
        CREATE TRIGGER messaging_node_identity BEFORE UPDATE ON messaging_nodes
        WHEN NEW.platform IS NOT OLD.platform OR NEW.chat_id IS NOT OLD.chat_id
            OR NEW.node_id IS NOT OLD.node_id OR NEW.root_id IS NOT OLD.root_id
            OR NEW.parent_id IS NOT OLD.parent_id OR NEW.parent_reference_id IS NOT OLD.parent_reference_id
        BEGIN SELECT RAISE(ABORT, 'Messaging topology is immutable'); END
    """)
    connection.execute("""
        CREATE TRIGGER messaging_node_references AFTER INSERT ON messaging_nodes
        BEGIN
            INSERT INTO messaging_references VALUES (NEW.platform, NEW.chat_id, NEW.node_id, NEW.node_id, 'prompt');
            INSERT INTO messaging_references SELECT NEW.platform, NEW.chat_id, NEW.status_message_id, NEW.node_id, 'status'
                WHERE NEW.status_message_id IS NOT NULL;
        END
    """)
    connection.execute("""
        CREATE TRIGGER messaging_status_reference AFTER UPDATE OF status_message_id ON messaging_nodes
        WHEN NEW.status_message_id IS NOT OLD.status_message_id
        BEGIN
            DELETE FROM messaging_references WHERE platform = OLD.platform AND chat_id = OLD.chat_id
                AND node_id = OLD.node_id AND kind = 'status';
            INSERT INTO messaging_references SELECT NEW.platform, NEW.chat_id, NEW.status_message_id, NEW.node_id, 'status'
                WHERE NEW.status_message_id IS NOT NULL;
        END
    """)
    connection.execute(
        "CREATE INDEX messaging_parent ON messaging_nodes(platform, chat_id, parent_id)"
    )
    connection.execute(
        "CREATE INDEX messaging_parent_reference ON messaging_nodes(platform, chat_id, parent_reference_id, parent_id)"
    )
    connection.execute("""
        CREATE TABLE messaging_managed_messages (
            sequence INTEGER PRIMARY KEY, platform TEXT NOT NULL, chat_id TEXT NOT NULL,
            message_id TEXT NOT NULL, ts TEXT NOT NULL,
            direction TEXT NOT NULL CHECK (direction IN ('in','out')),
            kind TEXT NOT NULL CHECK (length(kind) > 0),
            UNIQUE (platform, chat_id, message_id)
        )
    """)
    connection.execute(
        "CREATE INDEX messaging_message_order ON messaging_managed_messages(platform, chat_id, sequence)"
    )
    connection.execute("""
        CREATE TABLE messaging_legacy_import (
            source TEXT PRIMARY KEY NOT NULL CHECK (source = 'sessions.json'),
            outcome TEXT NOT NULL CHECK (outcome IN ('complete','partial','unreadable','absent')),
            trees INTEGER NOT NULL CHECK (trees >= 0), messages INTEGER NOT NULL CHECK (messages >= 0),
            skipped INTEGER NOT NULL CHECK (skipped >= 0),
            cleanup_pending INTEGER NOT NULL CHECK (cleanup_pending IN (0,1))
        )
    """)

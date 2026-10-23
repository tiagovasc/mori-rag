"""Public-source transport checks without contacting any website."""
import socket
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

from mori.ingest import _open_public_https, fetch_public


PUBLIC = (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))
PRIVATE = (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 443))


class FetchSecurityTests(unittest.TestCase):
    def test_transport_uses_approved_address_and_original_tls_hostname(self):
        raw, tls, connection = MagicMock(), MagicMock(), MagicMock()
        context = MagicMock()
        context.wrap_socket.return_value = tls
        with patch('mori.ingest.socket.socket', return_value=raw), \
             patch('mori.ingest.socket.getaddrinfo', side_effect=AssertionError('Unexpected DNS resolution')), \
             patch('mori.ingest.ssl.create_default_context', return_value=context), \
             patch('mori.ingest.http.client.HTTPSConnection', return_value=connection):
            result = _open_public_https(urlsplit('https://source.example/article?q=water#part'), [PUBLIC])
        raw.connect.assert_called_once_with(PUBLIC[4])
        context.wrap_socket.assert_called_once_with(raw, server_hostname='source.example')
        self.assertIs(connection.sock, tls)
        self.assertIs(result, connection)
        self.assertEqual(connection.request.call_args.args[:2], ('GET', '/article?q=water'))

    def test_mixed_public_private_dns_is_rejected_before_transport(self):
        with patch('mori.ingest.socket.getaddrinfo', return_value=[PUBLIC, PRIVATE]), \
             patch('mori.ingest._open_public_https') as connect:
            with self.assertRaisesRegex(ValueError, 'Private'):
                fetch_public('https://source.example/article')
        connect.assert_not_called()

    def test_redirect_to_private_target_stops_and_closes_first_connection(self):
        connection = MagicMock()
        response = connection.getresponse.return_value
        response.status = 302
        response.headers = {'Location': 'https://private.example/'}
        with patch('mori.ingest.socket.getaddrinfo', side_effect=[[PUBLIC], [PRIVATE]]), \
             patch('mori.ingest._open_public_https', return_value=connection) as connect:
            with self.assertRaisesRegex(ValueError, 'Private'):
                fetch_public('https://source.example/article')
        self.assertEqual(connect.call_count, 1)
        response.close.assert_called_once()
        connection.close.assert_called_once()

    def test_public_snapshot_preserves_text_and_closes_connection(self):
        connection = MagicMock()
        response = connection.getresponse.return_value
        response.status = 200
        response.headers = {'Content-Type': 'text/plain; charset=utf-8'}
        response.read.return_value = b'An explicit original source.'
        with patch('mori.ingest.socket.getaddrinfo', return_value=[PUBLIC]), \
             patch('mori.ingest._open_public_https', return_value=connection):
            units = fetch_public('https://source.example/article')
        self.assertEqual(units[0]['text'], 'An explicit original source.')
        self.assertEqual(units[0]['metadata']['url'], 'https://source.example/article')
        response.close.assert_called_once()
        connection.close.assert_called_once()

    def test_failed_tls_connection_closes_the_socket(self):
        raw, context = MagicMock(), MagicMock()
        context.wrap_socket.side_effect = OSError('Connection failure')
        with patch('mori.ingest.socket.socket', return_value=raw), \
             patch('mori.ingest.ssl.create_default_context', return_value=context):
            with self.assertRaisesRegex(ValueError, 'connection'):
                _open_public_https(urlsplit('https://source.example/article'), [PUBLIC])
        raw.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()

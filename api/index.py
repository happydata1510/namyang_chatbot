import os
import sys
import logging
from django.core.wsgi import get_wsgi_application

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set the Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'namyangju_SP.settings')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Get the WSGI application
    application = get_wsgi_application()
    logger.info("Django application loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Django application: {str(e)}")
    # Fallback application for debugging
    def application(environ, start_response):
        status = '500 Internal Server Error'
        response_headers = [('Content-Type', 'text/plain')]
        start_response(status, response_headers)
        return [f'Django application failed to load: {str(e)}'.encode()]

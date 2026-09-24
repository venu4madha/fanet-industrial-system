"""
Management command to create the predefined admin and operator accounts.
Run with: python manage.py create_default_users
"""
from django.core.management.base import BaseCommand
from accounts.models import CustomUser


class Command(BaseCommand):
    help = 'Create default admin and operator users for initial system access.'

    USERS = [
        {
            'username': 'admin',
            'password': 'Admin@123',
            'email': 'admin@fanet.system',
            'first_name': 'System',
            'last_name': 'Administrator',
            'role': 'admin',
            'department': 'IT Operations',
            'employee_id': 'EMP-ADM-001',
            'is_staff': True,
            'is_superuser': True,
        },
        {
            'username': 'operator',
            'password': 'Operator@123',
            'email': 'operator@fanet.system',
            'first_name': 'Line',
            'last_name': 'Operator',
            'role': 'user',
            'department': 'Quality Control',
            'employee_id': 'EMP-OPR-001',
            'is_staff': False,
            'is_superuser': False,
        },
    ]

    def handle(self, *args, **options):
        created = 0
        skipped = 0
        for data in self.USERS:
            password = data.pop('password')
            username = data['username']
            if CustomUser.objects.filter(username=username).exists():
                self.stdout.write(f'  [WARNING] User "{username}" already exists -- skipped.')
                skipped += 1
                data['password'] = password
                continue
            user = CustomUser(**data)
            user.set_password(password)
            user.save()
            self.stdout.write(
                self.style.SUCCESS(f'  [SUCCESS] Created user "{username}" ({data["role"]})')
            )
            data['password'] = password
            created += 1

        self.stdout.write('')
        self.stdout.write(f'Done: {created} created, {skipped} skipped.')
        if created + skipped > 0:
            self.stdout.write('')
            self.stdout.write('Credentials:')
            self.stdout.write('  Admin:    admin    / Admin@123')
            self.stdout.write('  Operator: operator / Operator@123')

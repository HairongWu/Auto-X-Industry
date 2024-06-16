import { MenuItem } from '../models/menu.model';

export class Menu {
  public static pages: MenuItem[] = [
    {
      group: 'Services',
      separator: true,
      items: [
        {
          icon: 'assets/icons/heroicons/outline/cog.svg',
          label: 'Recognize Anything',
          route: '/dashboard',
          children: [
            { label: 'Images/Videos', route: '/dashboard/img_video' },
            { label: 'Times Series', route: '/dashboard/ts' },
          ],
        },
        {
          icon: 'assets/icons/heroicons/outline/gift.svg',
          label: 'Document Intelligence',
          route: '/dashboard',
          children: [
            { label: 'Dashboard', route: '/dashboard/di' },
          ],
        },
      ],
    },
    {
      group: 'Account & Settings',
      separator: false,
      items: [
        {
          icon: 'assets/icons/heroicons/outline/lock-closed.svg',
          label: 'Auth',
          route: '/auth',
          children: [
            { label: 'Sign up', route: '/auth/sign-up' },
            { label: 'Sign in', route: '/auth/sign-in' },
            { label: 'Forgot Password', route: '/auth/forgot-password' },
            { label: 'New Password', route: '/auth/new-password' },
            { label: 'Two Steps', route: '/auth/two-steps' },
          ],
        },
        {
          icon: 'assets/icons/heroicons/outline/lock-closed.svg',
          label: 'Settings',
          route: '',
          children: [
            { label: 'Auto-X AI Server', route: '' },
          ],
        },
      ],
    },
  ];
}

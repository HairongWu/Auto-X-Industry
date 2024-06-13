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
        {
          icon: 'assets/icons/heroicons/outline/bookmark.svg',
          label: 'Auto-X Advisor',
          route: '',
          children: [
            { label: 'Lauch Advisor', route: '' },
            { label: 'Virtual Human Settings', route: '' },
          ],
        },     
        {
          icon: 'assets/icons/heroicons/outline/moon.svg',
          label: 'Auto Development',
          route: '',
          children: [
            { label: 'Modify Legacy Systems', route: '' },
            { label: 'Develop New Systems', route: '' },
          ],
        },   
        {
          icon: 'assets/icons/heroicons/outline/sun.svg',
          label: 'Auto Finance',
          route: '',
          children: [
            { label: 'Quantitative Trading', route: '' },
            { label: 'Data Source Settings', route: '' },
          ],
        },      
        {
          icon: 'assets/icons/heroicons/outline/chart-pie.svg',
          label: 'Auto Supply Chain',
          route: '',
          children: [
            { label: 'Demand forecasting', route: '' },
            { label: 'Inventory planning', route: '' },
            { label: 'Production planning and scheduling', route: '' },
            { label: 'Odoo and ERP integration', route: '' },
          ],
        },    
      ],
    },
    {
      group: 'Account',
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
      ],
    },
  ];
}

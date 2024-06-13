import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DashboardComponent } from './dashboard.component';
import { ImgVideoComponent } from './pages/img_video/img_video.component';
import { TSComponent } from './pages/ts/ts.component';
import { DIComponent } from './pages/di/di.component';

const routes: Routes = [
  {
    path: '',
    component: DashboardComponent,
    children: [
      { path: '', redirectTo: 'img_video', pathMatch: 'full' },
      { path: 'img_video', component: ImgVideoComponent },
      { path: 'ts', component: TSComponent },
      { path: 'di', component: DIComponent },
      { path: '**', redirectTo: 'errors/404' },
    ],
  },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class DashboardRoutingModule {}
